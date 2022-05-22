import matplotlib.pyplot as plt

from PIL import Image, ImageOps

og_image = Image.open(
    "../chess/test/1b1b2k1-K2B1q2-R3B2p-3b1NR1-5p2-3N4-8-5N2.jpeg")
og_image.show()

gray_image = ImageOps.grayscale(og_image)
gray_image.show()
