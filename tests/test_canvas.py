from shapes.canvas import Canvas


def test_canvas_without_target_should_contain_img_with_proper_number_of_channels():
    c = Canvas.without_target(size=100, background=None, channels=1)
    assert c.img.shape == (100, 100, 1)
    c = Canvas.without_target(size=100, background=None, channels=3)
    assert c.img.shape == (100, 100, 3)
