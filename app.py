import streamlit as st
from streamlit_drawable_canvas import st_canvas
from PIL import Image
import numpy as np
from assets.model import MobileNetV1
import torch
import time
import torch.nn.functional as F


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

mobilenetv1 = MobileNetV1(ch_in=1, n_classes=10)
mobilenetv1.load_state_dict(torch.load('assets/weights/MobileNetV1.pt', map_location=device))
mobilenetv1.eval()

def Code():
    st.header("Pytorch code for MobileNetV1")
    code = '''
    import torch.nn as nn
    class MobileNetV1(nn.Module):
        def __init__(self, ch_in, n_classes):
            super(MobileNetV1, self).__init__()

            def conv_bn(inp, oup, stride):
                return nn.Sequential(
                    nn.Conv2d(inp, oup, 3, stride, 1, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU(inplace=True)
                    )

            def conv_dw(inp, oup, stride):
                return nn.Sequential(
                    # dw
                    nn.Conv2d(inp, inp, 3, stride, 1, groups=inp, bias=False),
                    nn.BatchNorm2d(inp),
                    nn.ReLU(inplace=True),

                    # pw
                    nn.Conv2d(inp, oup, 1, 1, 0, bias=False),
                    nn.BatchNorm2d(oup),
                    nn.ReLU(inplace=True),
                )

            self.model = nn.Sequential(
                conv_bn(ch_in, 32, 2),
                conv_dw(32, 64, 1),
                conv_dw(64, 128, 2),
                conv_dw(128, 128, 1),
                conv_dw(128, 256, 2),
                conv_dw(256, 256, 1),
                conv_dw(256, 512, 2),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 512, 1),
                conv_dw(512, 1024, 2),
                conv_dw(1024, 1024, 1),
                nn.AdaptiveAvgPool2d(1)
            )
            self.fc = nn.Linear(1024, n_classes)

        def forward(self, x):
            x = self.model(x)
            x = x.view(-1, 1024)
            x = self.fc(x)
            return x
    '''
    st.code(code, language='python')


def Demo():    
    st.header("Demo on MNIST dataset")
    canvas_result = st_canvas(
        fill_color="rgba(0, 0, 0, 1)",
        stroke_width=15,
        stroke_color="#FFFFFF",
        background_color="#000000",
        height=280,
        width=280,
        drawing_mode="freedraw",
        key="canvas",
    )
    
    # Process the drawing when the user draws on the canvas
    if canvas_result.image_data is not None:
        img = Image.fromarray(canvas_result.image_data.astype('uint8'), 'RGBA')
        img = img.resize((28, 28)).convert('L')  # Convert to grayscale
        img_array = np.array(img)
        if len(np.unique(img_array)) != 1:

            # Display the processed 28x28 image
            st.write("Processed Image (28x28):")
            st.image(img, width=150)
            
            # Start timing the processing time
            start_time = time.time()

            # Convert the image to a tensor and send it to the appropriate device (CPU or GPU)
            img_tensor = torch.from_numpy(img_array).unsqueeze(0).unsqueeze(0).float().to(device) / 255.0

            # Predict the digit
            with torch.no_grad():
                out = mobilenetv1(img_tensor)

                # Apply softmax to get confidence scores
                probabilities = F.softmax(out, dim=1)

                # Get predicted class and its confidence score
                _, predicted = torch.max(probabilities, 1)
                confidence_score = probabilities[0][predicted.item()].item()

            # End timing the processing time
            end_time = time.time()
            # Calculate and display processing time
            process_time = end_time - start_time
            
            
            st.subheader(f"Predicted Digit: {predicted.item()}")
            st.subheader(f"Confidence: {confidence_score:.4f}")
            st.subheader(f"Processing Time: {process_time:.4f} seconds")

        else:
            st.write("Draw something on the canvas!")

pg = st.navigation([st.Page(Code), st.Page(Demo)])
pg.run()
