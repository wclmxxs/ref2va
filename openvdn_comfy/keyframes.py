"""Image roles and the native keyframe canvas preparation (no tensor dependencies)."""
MODES = {(): 't2va', ('first',): 'i2va', ('last',): 'l2va', ('first', 'last'): 'fl2va'}


def normalize_anchors(count, anchors=None):
    anchors = tuple(['ref'] * count if anchors is None else anchors)
    if len(anchors) != count or count > 9:
        raise ValueError('Each input image needs exactly one anchor; at most 9 images')
    if anchors and all(a == 'ref' for a in anchors):
        return anchors
    if anchors not in MODES:
        raise ValueError('Use references OR one first frame and/or one last frame, ordered first then last')
    return anchors


def conditioning_mode(anchors):
    return 'ref2va_like' if anchors and all(a == 'ref' for a in anchors) else MODES[tuple(anchors)]


def prepare_keyframes(images, width, height):
    """Match OpenVDN put_on_canvas, using the requested aligned generation canvas."""
    from PIL import Image
    prepared = []
    for index, image in enumerate(images):
        image = image.convert('RGB')
        if image.size == (width, height):
            prepared.append(image)
        elif index == 0:
            prepared.append(image.resize((width, height), Image.Resampling.LANCZOS))
        else:
            scale = max(width / image.width, height / image.height)
            size = (max(width, round(image.width * scale)), max(height, round(image.height * scale)))
            image = image.resize(size, Image.Resampling.LANCZOS)
            left, top = (size[0] - width) // 2, (size[1] - height) // 2
            prepared.append(image.crop((left, top, left + width, top + height)))
    return prepared
