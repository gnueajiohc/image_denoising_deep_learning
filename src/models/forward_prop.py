from .class_guided_unet import ClassGuidedUNet

def forward_prop(model, noisy_images):
        # 🟢 ClassGuidedUNet인지 확인 후 feature_output 추가
    if isinstance(model, ClassGuidedUNet):
        feature_output = model.classifier(noisy_images)  # Classifier 사용
        outputs = model(noisy_images, feature_output)  # 🟢 feature_output 전달
    else:
        outputs = model(noisy_images)
    
    return outputs