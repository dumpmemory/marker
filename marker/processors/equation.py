import re
from typing import Annotated, Tuple

from bs4 import BeautifulSoup
from ftfy import fix_text, TextFixerConfig
from surya.layout.schema import LayoutBox, LayoutResult
from surya.recognition import RecognitionPredictor

from marker.logger import get_logger
from marker.processors import BaseProcessor
from marker.schema import BlockTypes
from marker.schema.document import Document
from marker.schema.labels import block_type_to_surya_label

logger = get_logger()


class EquationProcessor(BaseProcessor):
    """
    A processor for recognizing equations (and other always-OCR blocks) on
    pages that kept their embedded pdftext text. OCR'd pages already have
    equation html from the OcrBuilder.
    """

    block_types: Annotated[
        Tuple[BlockTypes],
        "The block types to process.",
    ] = (BlockTypes.Equation, BlockTypes.ChemicalBlock)
    equation_token_budget: Annotated[
        int,
        "Token budget for recognizing a single equation block.",
    ] = 2000
    disable_tqdm: Annotated[
        bool,
        "Whether to disable the tqdm progress bar.",
    ] = False

    def __init__(self, recognition_model: RecognitionPredictor, config=None):
        super().__init__(config)

        self.recognition_model = recognition_model

    def __call__(self, document: Document):
        images = []
        layout_results = []
        block_ids = []

        for page in document.pages:
            target_blocks = [
                block
                for block in page.contained_blocks(document, self.block_types)
                if not block.html
            ]
            if not target_blocks:
                continue

            page_image = page.get_image(highres=True)
            page_size = page.polygon.size
            image_size = page_image.size

            boxes = []
            page_block_ids = []
            for i, block in enumerate(target_blocks):
                polygon = block.polygon.rescale(page_size, image_size).fit_to_bounds(
                    (0, 0, *image_size)
                )
                boxes.append(
                    LayoutBox(
                        polygon=polygon.polygon,
                        label=block_type_to_surya_label(block.block_type),
                        raw_label=str(block.block_type),
                        position=i,
                        count=block.layout_token_count or self.equation_token_budget,
                    )
                )
                page_block_ids.append(block.id)

            images.append(page_image)
            layout_results.append(
                LayoutResult(bboxes=boxes, image_bbox=[0, 0, *image_size])
            )
            block_ids.append(page_block_ids)

        if not images:
            return

        self.recognition_model.disable_tqdm = self.disable_tqdm
        recognition_results = self.recognition_model(
            images=images, layout_results=layout_results, full_page=False
        )

        for page_block_ids, page_result in zip(block_ids, recognition_results):
            assert len(page_block_ids) == len(page_result.blocks), (
                "Every equation block should have a corresponding prediction"
            )
            for block_id, block_result in zip(page_block_ids, page_result.blocks):
                if block_result.error or not block_result.html:
                    logger.warning(f"Equation recognition failed for {block_id}")
                    continue
                block = document.get_block(block_id)
                if block.block_type == BlockTypes.Equation:
                    block.html = self.fix_latex(block_result.html)
                else:
                    block.html = block_result.html.strip()

    def fix_latex(self, math_html: str):
        math_html = math_html.strip()
        soup = BeautifulSoup(math_html, "html.parser")
        opening_math_tag = soup.find("math")

        # No math block found
        if not opening_math_tag:
            return ""

        # Force block format
        opening_math_tag.attrs["display"] = "block"
        fixed_math_html = str(soup)

        # Sometimes model outputs newlines at the beginning/end of tags
        fixed_math_html = re.sub(
            r"^<math display=\"block\">\\n(?![a-zA-Z])",
            '<math display="block">',
            fixed_math_html,
        )
        fixed_math_html = re.sub(r"\\n</math>$", "</math>", fixed_math_html)
        fixed_math_html = re.sub(r"<br>", "", fixed_math_html)
        fixed_math_html = fix_text(
            fixed_math_html, config=TextFixerConfig(unescape_html=True)
        )
        return fixed_math_html
