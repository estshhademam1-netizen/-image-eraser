# ============================================================
# LAMA INPAINTING (Enhanced for Complete Object Removal)
# ============================================================

def run_lama(image, mask):
    model = get_lama()

    if model is None:
        raise RuntimeError(
            "LaMa is not available. "
            + str(lama_error)
        )

    # OpenCV BGR -> RGB
    rgb = cv2.cvtColor(
        image,
        cv2.COLOR_BGR2RGB
    )

    pil_image = Image.fromarray(
        rgb
    )

    # Mask: تأكد من أن القناع أبيض تماماً (255) للمنطقة المراد إزالتها وأسود (0) للباقي
    pil_mask = Image.fromarray(
        mask
    ).convert("L")

    print(
        "Running LaMa inpainting...",
        pil_image.size
    )

    # AI INPAINTING - ملء الفراغ وإزالة الكائن بالكامل
    result = model(
        pil_image,
        pil_mask
    )

    # PIL -> numpy
    result_np = np.array(
        result
    )

    # RGB -> BGR
    result_bgr = cv2.cvtColor(
        result_np,
        cv2.COLOR_RGB2BGR
    )

    return result_bgr


# ============================================================
# CLASSIC FALLBACK (Optimized to Prevent Pixelation)
# ============================================================

def run_classic(image, mask):
    # توسيع القناع قليلاً لضمان تغطية حواف الكائن بالكامل دون ترك تشويش
    kernel = np.ones((7, 7), np.uint8)
    dilated_mask = cv2.dilate(mask, kernel, iterations=2)

    # استخدام خوارزمية Navier-Stokes (NS) لملء الخلفية بشكل أنعم وأفضل من التغبيش
    result = cv2.inpaint(
        image,
        dilated_mask,
        inpaintRadius=5,
        flags=cv2.INPAINT_NS
    )

    return result
