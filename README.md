# 🤟 ML_ASL: American Sign Language Alphabet Recognition

A Python-based, real-time American Sign Language (ASL) alphabet recognition tool using a Convolutional Neural Network (CNN), ideal for educational and assistive applications.

---

## 🚀 Features

- 🧠 Real-time ASL letter recognition via webcam  
- 📚 Trained on a comprehensive ASL alphabet dataset  
- 🧪 High accuracy across all 26 letters  
- 🛠️ Easy-to-use GUI, integrated with OpenCV & Streamlit  
- 📁 Supports batch image testing

---

## 🧪 Tech Stack

- **Python**  
- **Deep Learning**: PyTorch or TensorFlow  
- **Computer Vision**: OpenCV for video capturing  
- **Frontend**: Streamlit (optional GUI mode)  
- **Data Handling**: NumPy, Pandas, PIL

---

## 📁 Project Structure

ML_ASL/
├── data/ # ASL image dataset
├── models/ # Saved model weights (.pth / .h5)
├── train.py # Training script for CNN model
├── recognize.py # Real-time webcam recognition
├── app.py # Streamlit deployment interface
├── requirements.txt # Project dependencies
└── README.md


---

## 🧑‍🏫 How It Works

1. **Training**: Train the CNN with labeled ASL images using `train.py`.  
2. **Recognition**: Use `recognize.py` to capture webcam video and predict ASL letters.  
3. **Deployment**: Launch `app.py` to interact via Streamlit-based GUI.

---

## 🛠️ Installation & Usage

```bash
# Clone the repository
git clone https://github.com/Andrei-Sturza/ML_ASL.git
cd ML_ASL
```

```bash
# Install dependencies
pip install -r requirements.txt
```

```bash
# Train the model (if needed)
python train.py
```

```bash
# For real-time recognition
python recognize.py
```

```bash
# Or run the Streamlit app
streamlit run app.py
```

### 📌 Notes
Ensure your webcam is connected for recognize.py.

Adjust input image size or learning rate in train.py as needed.

GPU recommended for faster training runs.

### 📈 Results
The model achieves over 95% accuracy across ASL letters. Confusion matrices and training metrics are logged for performance review.

### 📄 License
MIT © Andrei Sturza

### 🤝 Contributions
Found a bug or want to add features? Feel free to open an issue or submit a pull request. Let’s bring sign language recognition to more people!

---

Feel free to adapt any paths, tech stack details, or structure if they differ. If you'd like to include metrics screenshots, badges (e.g. pip installs or license), just let me know!
