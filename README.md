# QualityControl-PBA

<h4 align="center">
Braylon Trail, Jamie Boyd, Preston Harberts, Sebjin Kennedy, Guilherme Gripp
</h4>


This Hackathon project contains two powerful AI quality control tools to completely eliminate human error in tile manufacturing, bringing in automated efficiency and standardized consistency.

**Calibre Analysis (AI Tool 1)**
- We have trained a Convolutional Neural Network (CNN) to visually inspect every finished tile.
- The system determines the tile's exact calibre (categorized 3, 4, or 5).

**Label Automation (AI Tool 2)**
- We use computer vision to quickly read the existing product ID printed on the box.
- A new, correct label is automatically printed with all the shipment details.

Here are the tools we use:
- Frontend: React, HTML, CSS, JS, Hope UI, Bootstrap 
- Backend: JS, Python, CNNs, RNNs, OCR

### Backend

In the backend folder, run the following with Python:

```
pip install -r requirements.txt
```

Or

```
pip install fastapi==0.119.0 uvicorn==0.37.0 torch==2.9.0 torchvision==0.24.0 paddleocr==3.3.0 paddlepaddle==3.2.0 opencv-python==4.12.0.88 pillow==12.0.0 numpy==2.2.6 pydantic==2.12.2 httpx==0.28.1 scipy==1.16.2 scikit-image==0.25.2 pandas==2.3.3
```

### Frontend

In the frontend folder, run the following with NPM:

```
npm install
```

Start the website with this:

```
npm start
```

Then open `http://localhost:3000/` in a browser.
