from dataclasses import dataclass
from io import BytesIO

import pypdfium2 as pdfium
from PIL import Image, ImageOps, UnidentifiedImageError


IMAGE_FORMATS = {
    ".jpg": "JPEG",
    ".jpeg": "JPEG",
    ".png": "PNG",
}
PDF_RENDER_SCALE = 2
MAX_UPLOAD_SIZE_BYTES = 10 * 1024 * 1024
MAX_PDF_PAGES = 20
MAX_IMAGE_PIXELS = 25_000_000
MAX_TOTAL_RENDERED_PIXELS = 100_000_000


@dataclass
class PrescriptionPage:
    page_number: int
    image: Image.Image | None
    error: str | None = None
    selectable_text: str | None = None
    has_selectable_text: bool = False


class UploadValidationError(ValueError):
    pass


def load_prescription_pages(filename: str, content: bytes) -> list[PrescriptionPage]:
    suffix = filename.rsplit(".", 1)[-1].lower() if "." in filename else ""
    suffix = f".{suffix}"

    if not content:
        raise UploadValidationError("The uploaded file is empty.")
    if len(content) > MAX_UPLOAD_SIZE_BYTES:
        raise UploadValidationError(
            "The uploaded file exceeds the 10 MB limit."
        )

    if suffix in IMAGE_FORMATS:
        return [_load_image_page(content, suffix)]

    if suffix == ".pdf":
        return _load_pdf_pages(content)

    raise UploadValidationError(
        "Unsupported file type. Upload a JPG, JPEG, PNG, or PDF file."
    )


def _load_image_page(content: bytes, suffix: str) -> PrescriptionPage:
    try:
        with Image.open(BytesIO(content)) as source:
            detected_format = source.format
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise UploadValidationError(
                    "The uploaded image dimensions exceed the processing limit."
                )
            source.verify()
    except UploadValidationError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ):
        raise UploadValidationError(
            "The uploaded image could not be decoded."
        ) from None

    if detected_format != IMAGE_FORMATS[suffix]:
        raise UploadValidationError(
            "The file contents do not match the selected image format."
        )

    try:
        with Image.open(BytesIO(content)) as source:
            if source.width * source.height > MAX_IMAGE_PIXELS:
                raise UploadValidationError(
                    "The uploaded image dimensions exceed the processing limit."
                )
            image = ImageOps.exif_transpose(source).convert("RGB")
            image.load()
    except UploadValidationError:
        raise
    except (
        UnidentifiedImageError,
        OSError,
        ValueError,
        Image.DecompressionBombError,
    ):
        raise UploadValidationError(
            "The uploaded image could not be decoded."
        ) from None

    return PrescriptionPage(page_number=1, image=image.copy())


def _load_pdf_pages(content: bytes) -> list[PrescriptionPage]:
    if b"%PDF-" not in content[:1024]:
        raise UploadValidationError("The uploaded file is not a valid PDF.")

    try:
        document = pdfium.PdfDocument(content)
    except (OSError, RuntimeError, ValueError):
        raise UploadValidationError("The uploaded PDF could not be opened.") from None

    try:
        try:
            page_count = len(document)
        except (OSError, RuntimeError, ValueError):
            raise UploadValidationError(
                "The uploaded PDF pages could not be read."
            ) from None

        if page_count == 0:
            raise UploadValidationError("The uploaded PDF contains no pages.")
        if page_count > MAX_PDF_PAGES:
            raise UploadValidationError(
                f"The uploaded PDF exceeds the {MAX_PDF_PAGES}-page limit."
            )

        pages = []
        total_rendered_pixels = 0
        for index in range(page_count):
            pdf_page = None
            bitmap = None
            textpage = None
            try:
                pdf_page = document[index]
                width, height = pdf_page.get_size()
                rendered_pixels = (
                    int(width * PDF_RENDER_SCALE) * int(height * PDF_RENDER_SCALE)
                )
                if rendered_pixels <= 0 or rendered_pixels > MAX_IMAGE_PIXELS:
                    raise UploadValidationError(
                        "A PDF page exceeds the image processing limit."
                    )
                total_rendered_pixels += rendered_pixels
                if total_rendered_pixels > MAX_TOTAL_RENDERED_PIXELS:
                    raise UploadValidationError(
                        "The PDF exceeds the total processing limit."
                    )
                bitmap = pdf_page.render(scale=PDF_RENDER_SCALE)
                image = bitmap.to_pil().convert("RGB").copy()

                # Check for selectable text
                selectable_text = None
                has_selectable_text = False
                try:
                    textpage = pdf_page.get_textpage()
                    extracted = textpage.get_text_range()
                    if extracted and len(extracted.strip()) > 5:
                        selectable_text = extracted.strip()
                        has_selectable_text = True
                except Exception:
                    pass

                pages.append(
                    PrescriptionPage(
                        page_number=index + 1,
                        image=image,
                        selectable_text=selectable_text,
                        has_selectable_text=has_selectable_text,
                    )
                )
            except UploadValidationError:
                raise
            except (OSError, RuntimeError, ValueError, IndexError):
                pages.append(
                    PrescriptionPage(
                        page_number=index + 1,
                        image=None,
                        error="This PDF page could not be read.",
                    )
                )
            finally:
                if textpage is not None:
                    try:
                        textpage.close()
                    except Exception:
                        pass
                if bitmap is not None:
                    bitmap.close()
                if pdf_page is not None:
                    pdf_page.close()

        return pages
    finally:
        document.close()
