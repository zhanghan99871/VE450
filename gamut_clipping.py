import io
from PIL import Image, ImageCms


class GamutClipper:

    class ClipIntent():
        GAMUT_CLIPPING = 0
        MAINTAIN_H = 1
        MAINTAIN_LH = 2

        def __init__(self):
            self.option = self.GAMUT_CLIPPING

    def __get_intent(
        clip_intent: ClipIntent
    ) -> ImageCms.ImageCmsTransform:
        rendering_intent = None
        if clip_intent == GamutClipper.ClipIntent.GAMUT_CLIPPING:
            rendering_intent = ImageCms.Intent.ABSOLUTE_COLORIMETRIC
        elif clip_intent == GamutClipper.ClipIntent.MAINTAIN_H:
            rendering_intent = ImageCms.Intent.PERCEPTUAL
        elif clip_intent == GamutClipper.ClipIntent.MAINTAIN_LH:
            rendering_intent = ImageCms.Intent.RELATIVE_COLORIMETRIC
        return rendering_intent

    def clip(
        im: Image, clip_intent: ClipIntent,
        output_icc_path: str = None
    ) -> Image:
        # if there is no colorspace info in the image, directly return it
        if im.info.get("icc_profile") is None:
            return im

        # get the input icc profile
        input_icc_profile = io.BytesIO(im.info.get('icc_profile'))
        input_icc = ImageCms.ImageCmsProfile(input_icc_profile)
        # get the output icc profile
        output_icc = ImageCms.getOpenProfile(output_icc_path)
        # get rendering intent from cliping intent
        rendering_intent = GamutClipper.__get_intent(clip_intent)

        map = ImageCms.ImageCmsTransform(
            input=input_icc,
            output=output_icc,
            input_mode="RGB",
            output_mode="RGB",
            intent=rendering_intent
        )
        return ImageCms.applyTransform(im, map)


# test code
if __name__ == "__main__":
    image = Image.open("data/VW310-6CS.DRL-20220328.HV_001.png")
    # output_icc_path = "/usr/share/color/icc/colord/AdobeRGB1998.icc"
    output_icc_path = "C:\Windows\System32\spool\drivers\color\sRGB Color Space Profile.icm"
    clip_intent = GamutClipper.ClipIntent.GAMUT_CLIPPING
    clipped_image = GamutClipper.clip(image, clip_intent, output_icc_path)
    clipped_image.show()
