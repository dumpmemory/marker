import re
from collections import defaultdict, Counter
from copy import deepcopy
from typing import Annotated, List, Literal, Optional

from bs4 import BeautifulSoup
from ftfy import fix_text
from pydantic import BaseModel
from surya.recognition import _detect_repeat_loop
from surya.table_rec import TableRecPredictor
from surya.table_rec.schema import TableResult
from pdftext.extraction import table_output

from marker.processors import BaseProcessor
from marker.schema import BlockTypes
from marker.schema.blocks.tablecell import TableCell
from marker.schema.document import Document
from marker.schema.polygon import PolygonBox
from marker.util import matrix_intersection_area, unwrap_math
from marker.logger import get_logger

logger = get_logger()


class MarkerTableCell(BaseModel):
    """Mutable cell used during table assembly. The new surya simple-mode
    cells are bare geometry; spanning/header info is added marker-side."""

    polygon: List[List[float]]
    row_id: int
    col_id: int
    cell_id: int
    rowspan: int = 1
    colspan: int = 1
    is_header: bool = False
    within_row_id: int = 0
    text_lines: Optional[list] = None

    @classmethod
    def from_bbox(cls, bbox: List[float], **kwargs) -> "MarkerTableCell":
        x0, y0, x1, y1 = bbox
        return cls(polygon=[[x0, y0], [x1, y0], [x1, y1], [x0, y1]], **kwargs)

    @property
    def bbox(self) -> List[float]:
        xs = [p[0] for p in self.polygon]
        ys = [p[1] for p in self.polygon]
        return [min(xs), min(ys), max(xs), max(ys)]


class TableProcessor(BaseProcessor):
    """
    A processor for recognizing tables in the document.

    Hybrid strategy: tables with good embedded pdftext use the simple
    table-rec mode (rows/cols from the VLM, cell text from pdftext); tables
    that need OCR use the full mode, where the VLM emits the complete table
    HTML in one pass.
    """

    block_types = (BlockTypes.Table, BlockTypes.TableOfContents, BlockTypes.Form)
    table_rec_mode: Annotated[
        Literal["hybrid", "simple", "full"],
        "Table recognition strategy. 'hybrid' uses simple mode for tables with good",
        "embedded text and full (HTML) mode for tables needing OCR.",
    ] = "hybrid"
    simple_fallback_to_full: Annotated[
        bool,
        "Re-run degenerate simple-mode results through full (HTML) mode.",
    ] = True
    full_mode_token_floor: Annotated[
        int,
        "Minimum token budget for full-mode table recognition.",
    ] = 2048
    simple_mode_first_row_header: Annotated[
        bool,
        "Mark the first row of simple-mode tables as a header row.",
    ] = True
    contained_block_types: Annotated[
        List[BlockTypes],
        "Block types to remove if they're contained inside the tables.",
    ] = (BlockTypes.Text, BlockTypes.TextInlineMath)
    row_split_threshold: Annotated[
        float,
        "The percentage of rows that need to be split across the table before row splitting is active.",
    ] = 0.5
    pdftext_workers: Annotated[
        int,
        "The number of workers to use for pdftext.",
    ] = 1
    disable_tqdm: Annotated[
        bool,
        "Whether to disable the tqdm progress bar.",
    ] = False
    disable_ocr: Annotated[bool, "Disable OCR entirely."] = False

    def __init__(
        self,
        table_rec_model: TableRecPredictor,
        config=None,
    ):
        super().__init__(config)

        self.table_rec_model = table_rec_model
        # Conversion stats, useful for monitoring table rec quality
        self.table_stats = Counter()

    def __call__(self, document: Document):
        self.table_rec_model.disable_tqdm = self.disable_tqdm

        table_data, tables_by_page = self.collect_tables(document)
        if not table_data:
            return

        # Get pdftext cell text for tables on pages with good embedded text.
        # Tables where pdftext finds nothing are flipped to ocr_block before
        # mode routing.
        extract_blocks = [t for t in table_data if not t["ocr_block"]]
        self.assign_pdftext_lines(document, extract_blocks)

        simple_data, full_data = self.partition_tables(table_data)
        self.run_simple_mode(simple_data, full_data)
        self.run_full_mode(document, full_data)

        self.assemble_cells(document, simple_data)
        self.cleanup_contained_blocks(document, tables_by_page)

        # Release the cached raw pdftext pages - they hold char-level data
        for page in document.pages:
            page.pdftext_page = None

        self.table_stats["tables_total"] = len(table_data)
        logger.info(f"Table processing stats: {dict(self.table_stats)}")

    def collect_tables(self, document: Document):
        table_data = []
        tables_by_page = {}
        for page in document.pages:
            page_tables = page.contained_blocks(document, self.block_types)
            tables_by_page[page.page_id] = page_tables
            if not page_tables:
                continue

            highres_image = page.get_image(highres=True)
            image_size = highres_image.size
            page_size = page.polygon.size

            for block in page_tables:
                if block.block_type == BlockTypes.Table:
                    block.polygon = block.polygon.expand(0.01, 0.01)
                image_poly = block.polygon.rescale(page_size, image_size)
                table_image = highres_image.crop(image_poly.bbox)

                table_data.append(
                    {
                        "block_id": block.id,
                        "page_id": page.page_id,
                        "table_image": table_image,
                        "table_bbox": image_poly.bbox,
                        "img_size": image_size,
                        "token_count": block.layout_token_count or 0,
                        "ocr_block": any(
                            [
                                page.text_extraction_method == "surya",
                                page.ocr_errors_detected,
                            ]
                        ),
                    }
                )
        return table_data, tables_by_page

    def partition_tables(self, table_data: list):
        simple_data = []
        full_data = []
        for entry in table_data:
            if self.table_rec_mode == "full":
                use_full = True
            elif self.table_rec_mode == "simple" or self.disable_ocr:
                use_full = False
            else:  # hybrid
                use_full = entry["ocr_block"]

            if use_full:
                full_data.append(entry)
            else:
                simple_data.append(entry)
        return simple_data, full_data

    def run_simple_mode(self, simple_data: list, full_data: list):
        """Run simple-mode table rec, assign pdftext text to the geometric
        cells, and apply marker's cell postprocessing. Degenerate results are
        re-routed to full mode."""
        if not simple_data:
            return

        results: List[TableResult] = self.table_rec_model.predict_simple(
            [t["table_image"] for t in simple_data]
        )
        assert len(results) == len(simple_data), (
            "Number of table results should match the number of tables"
        )

        for entry, result in zip(simple_data, results):
            entry["cells"] = [
                MarkerTableCell(
                    polygon=cell.polygon,
                    row_id=cell.row_id,
                    col_id=cell.col_id,
                    cell_id=cell.cell_id,
                    is_header=self.simple_mode_first_row_header and cell.row_id == 0,
                )
                for cell in result.cells
            ]
            entry["unassigned_frac"] = self.assign_text_to_cells(entry)

            degenerate = result.error or len(entry["cells"]) == 0
            mostly_unassigned = (
                not entry["ocr_block"] and entry.get("unassigned_frac", 0) > 0.5
            )
            if degenerate or mostly_unassigned:
                self.table_stats["tables_simple_degenerate"] += 1
                if self.simple_fallback_to_full and not self.disable_ocr:
                    entry["cells"] = None
                    full_data.append(entry)
                continue

            self.table_stats["tables_simple"] += 1

        # Drop entries that were re-routed to full mode
        simple_data[:] = [t for t in simple_data if t.get("cells") is not None]

        self.split_combined_rows(simple_data)
        self.combine_dollar_column(simple_data)

    def run_full_mode(self, document: Document, full_data: list):
        """Full-mode table rec: the VLM emits the complete table HTML, which
        is set directly on the block."""
        if not full_data:
            return

        remaining = full_data
        for attempt in range(2):
            results = self.table_rec_model.predict_full(
                [t["table_image"] for t in remaining],
                counts=[
                    max(t["token_count"], self.full_mode_token_floor) for t in remaining
                ],
            )
            failed = []
            for entry, result in zip(remaining, results):
                html = self.clean_table_html(result.html if not result.error else "")
                if not html:
                    failed.append(entry)
                    continue

                block = document.get_block(entry["block_id"])
                block.structure = []
                block.html = html
                block.text_extraction_method = "surya"
                self.table_stats["tables_full"] += 1

            if not failed:
                break
            remaining = failed
        else:
            for entry in remaining:
                self.table_stats["tables_full_failed"] += 1
                logger.warning(
                    f"Full-mode table recognition failed for block {entry['block_id']}"
                )

    def clean_table_html(self, html: str | None) -> str:
        if not html:
            return ""
        if "<table" not in html:
            return ""
        if _detect_repeat_loop(html):
            return ""

        # Re-serialize to balance tags in case of token-budget truncation
        soup = BeautifulSoup(html, "html.parser")
        table = soup.find("table")
        if table is None or not table.find_all(["td", "th"]):
            return ""
        return str(soup).strip()

    def assemble_cells(self, document: Document, simple_data: list):
        """Convert assembled marker cells into TableCell blocks on the page."""
        for entry in simple_data:
            block = document.get_block(entry["block_id"])
            page = document.get_page(entry["page_id"])
            image_size = entry["img_size"]
            page_size = page.polygon.size

            block.structure = []  # Remove any existing lines, spans, etc.
            for cell in entry["cells"]:
                # Rescale the cell polygon to the page size
                cell_polygon = PolygonBox(polygon=cell.polygon).rescale(
                    image_size, page_size
                )

                # Rescale cell polygon to be relative to the page instead of the table
                for corner in cell_polygon.polygon:
                    corner[0] += block.polygon.bbox[0]
                    corner[1] += block.polygon.bbox[1]

                cell_block = TableCell(
                    polygon=cell_polygon,
                    text_lines=self.finalize_cell_text(cell),
                    rowspan=cell.rowspan,
                    colspan=cell.colspan,
                    row_id=cell.row_id,
                    col_id=cell.col_id,
                    is_header=bool(cell.is_header),
                    page_id=page.page_id,
                )
                page.add_full_block(cell_block)
                block.add_structure(cell_block)

    def cleanup_contained_blocks(self, document: Document, tables_by_page: dict):
        # Clean out other blocks inside the table
        # This can happen with stray text blocks inside the table post-merging
        for page in document.pages:
            page_tables = tables_by_page.get(page.page_id, [])
            if not page_tables:
                continue
            child_contained_blocks = page.contained_blocks(
                document, self.contained_block_types
            )
            if not child_contained_blocks:
                continue

            intersections = matrix_intersection_area(
                [c.polygon.bbox for c in child_contained_blocks],
                [block.polygon.bbox for block in page_tables],
            )
            for child_idx, child in enumerate(child_contained_blocks):
                for table_idx in range(len(page_tables)):
                    intersection_pct = intersections[child_idx, table_idx] / max(
                        child.polygon.area, 1
                    )
                    if intersection_pct > 0.95 and child.id in page.structure:
                        page.structure.remove(child.id)
                        break

    def finalize_cell_text(self, cell: MarkerTableCell):
        fixed_text = []
        text_lines = cell.text_lines if cell.text_lines else []
        for line in text_lines:
            text = line["text"].strip()
            if not text or text == ".":
                continue
            # Spaced sequences: ". . .", "- - -", "_ _ _", "… … …"
            text = re.sub(r"(\s?[.\-_…]){2,}", "", text)
            # Unspaced sequences: "...", "---", "___", "……"
            text = re.sub(r"[.\-_…]{2,}", "", text)
            # Remove mathbf formatting if there is only digits with decimals/commas/currency symbols inside
            text = re.sub(r"\\mathbf\{([0-9.,$€£]+)\}", r"<b>\1</b>", text)
            # Drop empty tags like \overline{}
            text = re.sub(r"\\[a-zA-Z]+\{\s*\}", "", text)
            # Drop \phantom{...} (remove contents too)
            text = re.sub(r"\\phantom\{.*?\}", "", text)
            # Drop \quad
            text = re.sub(r"\\quad", "", text)
            # Drop \,
            text = re.sub(r"\\,", "", text)
            # Unwrap \mathsf{...}
            text = re.sub(r"\\mathsf\{([^}]*)\}", r"\1", text)
            # Handle unclosed tags: keep contents, drop the command
            text = re.sub(r"\\[a-zA-Z]+\{([^}]*)$", r"\1", text)
            # If the whole string is \text{...} → unwrap
            text = re.sub(r"^\s*\\text\{([^}]*)\}\s*$", r"\1", text)

            # In case the above steps left no more latex math - We can unwrap
            text = unwrap_math(text)
            text = self.normalize_spaces(fix_text(text))
            fixed_text.append(text)
        return fixed_text

    @staticmethod
    def normalize_spaces(text):
        space_chars = [
            " ",  # em space
            " ",  # en space
            " ",  # non-breaking space
            "​",  # zero-width space
            "　",  # ideographic space
        ]
        for space in space_chars:
            text = text.replace(space, " ")
        return text

    def combine_dollar_column(self, simple_data: list):
        for entry in simple_data:
            cells = entry["cells"]
            if len(cells) == 0:
                # Skip empty tables
                continue
            unique_cols = sorted(list(set([c.col_id for c in cells])))
            max_col = max(unique_cols)
            dollar_cols = []
            for col in unique_cols:
                # Cells in this col
                col_cells = [c for c in cells if c.col_id == col]
                # Cheap raw-text pre-check before the expensive regex pipeline:
                # an all-dollar column must have only ""/"$" raw cell text.
                raw_text = [
                    "".join(line["text"] for line in (c.text_lines or [])).strip()
                    for c in col_cells
                ]
                if not all(rt in ("", "$") for rt in raw_text):
                    continue
                col_text = [
                    "\n".join(self.finalize_cell_text(c)).strip() for c in col_cells
                ]
                all_dollars = all([ct in ["", "$"] for ct in col_text])
                colspans = [c.colspan for c in col_cells]
                span_into_col = [
                    c
                    for c in cells
                    if c.col_id != col and c.col_id + c.colspan > col > c.col_id
                ]

                # This is a column that is entirely dollar signs
                if all(
                    [
                        all_dollars,
                        len(col_cells) > 1,
                        len(span_into_col) == 0,
                        all([c == 1 for c in colspans]),
                        col < max_col,
                    ]
                ):
                    next_col_cells = [c for c in cells if c.col_id == col + 1]
                    next_col_rows = [c.row_id for c in next_col_cells]
                    col_rows = [c.row_id for c in col_cells]
                    if (
                        len(next_col_cells) == len(col_cells)
                        and next_col_rows == col_rows
                    ):
                        dollar_cols.append(col)

            if len(dollar_cols) == 0:
                continue

            dollar_cols = sorted(dollar_cols)
            col_offset = 0
            for col in unique_cols:
                col_cells = [c for c in cells if c.col_id == col]
                if col_offset == 0 and col not in dollar_cols:
                    continue

                if col in dollar_cols:
                    col_offset += 1
                    for cell in col_cells:
                        text_lines = cell.text_lines if cell.text_lines else []
                        next_row_col = [
                            c
                            for c in cells
                            if c.row_id == cell.row_id and c.col_id == col + 1
                        ]

                        # Add dollar to start of the next column
                        next_text_lines = (
                            next_row_col[0].text_lines
                            if next_row_col[0].text_lines
                            else []
                        )
                        next_row_col[0].text_lines = deepcopy(text_lines) + deepcopy(
                            next_text_lines
                        )
                        cells[:] = [
                            c for c in cells if c.cell_id != cell.cell_id
                        ]  # Remove original cell
                        next_row_col[0].col_id -= col_offset
                else:
                    for cell in col_cells:
                        cell.col_id -= col_offset
            entry["cells"] = cells

    def split_combined_rows(self, simple_data: list):
        for entry in simple_data:
            cells = entry["cells"]
            if len(cells) == 0:
                # Skip empty tables
                continue
            unique_rows = sorted(list(set([c.row_id for c in cells])))
            row_info = []
            for row in unique_rows:
                # Cells in this row. References for now - only read here; we
                # deepcopy below (after the split threshold passes) to guard
                # the in-place row_id mutation, so non-splitting tables (the
                # common case) skip the copy entirely.
                row_cells = [c for c in cells if c.row_id == row]
                rowspans = [c.rowspan for c in row_cells]
                line_lens = [
                    len(c.text_lines) if isinstance(c.text_lines, list) else 1
                    for c in row_cells
                ]

                # Other cells that span into this row
                rowspan_cells = [
                    c
                    for c in cells
                    if c.row_id != row and c.row_id + c.rowspan > row > c.row_id
                ]
                should_split_entire_row = all(
                    [
                        len(row_cells) > 1,
                        len(rowspan_cells) == 0,
                        all([rowspan == 1 for rowspan in rowspans]),
                        all([line_len > 1 for line_len in line_lens]),
                        all([line_len == line_lens[0] for line_len in line_lens]),
                    ]
                )
                line_lens_counter = Counter(line_lens)
                counter_keys = sorted(list(line_lens_counter.keys()))
                should_split_partial_row = all(
                    [
                        len(row_cells) > 3,  # Only split if there are more than 3 cells
                        len(rowspan_cells) == 0,
                        all([r == 1 for r in rowspans]),
                        len(line_lens_counter) == 2
                        and counter_keys[0] <= 1
                        and counter_keys[1] > 1
                        and line_lens_counter[counter_keys[0]]
                        == 1,  # Allow a single column with a single line - keys are the line lens, values are the counts
                    ]
                )
                should_split = should_split_entire_row or should_split_partial_row
                row_info.append(
                    {
                        "should_split": should_split,
                        "row_cells": row_cells,
                        "line_lens": line_lens,
                    }
                )

            # Don't split if we're not splitting most of the rows in the table.  This avoids splitting stray multiline rows.
            if (
                sum([r["should_split"] for r in row_info]) / len(row_info)
                < self.row_split_threshold
            ):
                continue

            # We're going to split (and mutate row_id on non-split rows below),
            # so copy now to avoid aliasing the original cells.
            for item_info in row_info:
                item_info["row_cells"] = deepcopy(item_info["row_cells"])

            new_cells = []
            shift_up = 0
            max_cell_id = max([c.cell_id for c in cells])
            new_cell_count = 0
            for row, item_info in zip(unique_rows, row_info):
                max_lines = max(item_info["line_lens"])
                if item_info["should_split"]:
                    for i in range(0, max_lines):
                        for cell in item_info["row_cells"]:
                            # Calculate height based on number of splits
                            split_height = cell.bbox[3] - cell.bbox[1]
                            current_bbox = [
                                cell.bbox[0],
                                cell.bbox[1] + i * split_height,
                                cell.bbox[2],
                                cell.bbox[1] + (i + 1) * split_height,
                            ]

                            line = (
                                [cell.text_lines[i]]
                                if cell.text_lines and i < len(cell.text_lines)
                                else None
                            )
                            cell_id = max_cell_id + new_cell_count
                            new_cells.append(
                                MarkerTableCell.from_bbox(
                                    current_bbox,
                                    text_lines=line,
                                    rowspan=1,
                                    colspan=cell.colspan,
                                    row_id=cell.row_id + shift_up + i,
                                    col_id=cell.col_id,
                                    is_header=cell.is_header
                                    and i == 0,  # Only first line is header
                                    within_row_id=cell.within_row_id,
                                    cell_id=cell_id,
                                )
                            )
                            new_cell_count += 1

                    # For each new row we add, shift up subsequent rows
                    # The max is to account for partial rows
                    shift_up += max_lines - 1
                else:
                    for cell in item_info["row_cells"]:
                        cell.row_id += shift_up
                        new_cells.append(cell)

            # Only update the cells if we added new cells
            if len(new_cells) > len(cells):
                entry["cells"] = new_cells

    def assign_text_to_cells(self, entry: dict) -> float:
        """Assign pdftext lines to geometric cells. Returns the fraction of
        text lines that could not be assigned to any cell."""
        table_text_lines = entry.get("table_text_lines") or []
        table_cells: List[MarkerTableCell] = entry["cells"]
        if not table_cells:
            # No cells: every text line is unassigned (1.0), or nothing to do (0.0)
            return 1.0 if table_text_lines else 0.0
        if not table_text_lines:
            return 0.0

        text_line_bboxes = [t["bbox"] for t in table_text_lines]
        table_cell_bboxes = [c.bbox for c in table_cells]

        intersection_matrix = matrix_intersection_area(
            text_line_bboxes, table_cell_bboxes
        )

        unassigned = 0
        cell_text = defaultdict(list)
        for text_line_idx, table_text_line in enumerate(table_text_lines):
            intersections = intersection_matrix[text_line_idx]
            if intersections.sum() == 0:
                unassigned += 1
                continue

            max_intersection = intersections.argmax()
            cell_text[max_intersection].append(table_text_line)

        for k in cell_text:
            # TODO: see if the text needs to be sorted (based on rotation)
            text = cell_text[k]
            assert all("text" in t for t in text), "All text lines must have text"
            assert all("bbox" in t for t in text), "All text lines must have a bbox"
            table_cells[k].text_lines = text

        return unassigned / len(table_text_lines)

    def assign_pdftext_lines(self, document: Document, extract_blocks: list):
        if not extract_blocks:
            return

        # Group tables by page in one pass (preserves per-page order)
        blocks_by_page = defaultdict(list)
        for block in extract_blocks:
            blocks_by_page[block["page_id"]].append(block)
        unique_pages = list(blocks_by_page.keys())

        table_inputs = [
            {
                "tables": [b["table_bbox"] for b in page_blocks],
                "img_size": page_blocks[0]["img_size"],  # same for all on a page
            }
            for page_blocks in (blocks_by_page[p] for p in unique_pages)
        ]

        # Use the raw pdftext pages cached by the provider when available -
        # this avoids re-opening and re-extracting the PDF.
        cached_pages = [
            document.get_page(page_id).pdftext_page for page_id in unique_pages
        ]
        if any(p is None for p in cached_pages):
            cached_pages = None

        cell_text = table_output(
            document.filepath,
            table_inputs,
            page_range=unique_pages,
            workers=self.pdftext_workers,
            pages=cached_pages,
        )
        assert len(cell_text) == len(unique_pages), (
            "Number of pages and table inputs must match"
        )

        for page_tables, pnum in zip(cell_text, unique_pages):
            page_blocks = blocks_by_page[pnum]
            assert len(page_tables) == len(page_blocks), (
                "Number of tables and table inputs must match"
            )
            for block, table_text in zip(page_blocks, page_tables):
                if len(table_text) == 0:
                    # Re-OCR the block if pdftext didn't find any text
                    block["ocr_block"] = True
                else:
                    block["table_text_lines"] = table_text
