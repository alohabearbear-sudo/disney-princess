import streamlit as st
import torch
import torch.nn as nn
from torchvision import models, transforms
from PIL import Image
import os
import urllib.request

# ==========================================
# 👑 1. 設定頁面與 11 類公主標籤 (完全符合系統排序)
# ==========================================
st.set_page_config(page_title="迪士尼公主 AI 辨識系統", page_icon="👑", layout="centered")

st.title("👑 迪士尼公主 AI 影像辨識系統")
st.write("上傳一張迪士尼公主的圖片，讓 ResNet-50 模型幫你辨識她是誰！")

# ⚠️ 系統中大寫排前面、小寫排後面的絕對順序
CLASS_NAMES = [
    "Cinderella (灰姑娘)", 
    "Snow White (白雪公主)", 
    "Anna (安娜)", 
    "Ariel (小美人魚)", 
    "Aurora (睡美人/arura)", 
    "Belle (貝兒)", 
    "Elsa (艾莎)", 
    "Jasmine (茉莉公主)", 
    "Merida (梅莉達)", 
    "Rapunzel (樂佩/ruponzel)", 
    "Tiana (蒂安娜)"
]

# ==========================================
# ⚙️ 2. 載入模型權重 (支援 GitHub 自動下載)
# ==========================================
@st.cache_resource
def load_princess_model():
    num_classes = len(CLASS_NAMES)
    weights_path = "resnet50_fold_1_best.pth"
    # 你的 GitHub Release 直鏈
    download_url = "https://github.com/alohabearbear-sudo/disney-princess/releases/download/v1/resnet50_fold_1_best.pth"
    
    # 建立與訓練時完全相同的 ResNet-50 架構
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # 檢查本地是否存在權重，若無則啟動自動下載
    if not os.path.exists(weights_path):
        with st.spinner("📥 正在從 GitHub 下載雲端模型權重（僅需下載一次，請稍候...）"):
            try:
                urllib.request.urlretrieve(download_url, weights_path)
                st.toast("🎉 模型權重下載成功！", icon="✅")
            except Exception as e:
                st.error(f"❌ 從 GitHub 下載權重失敗: {e}")
                return None
                
    # 載入權重 (強制對應到 CPU，確保在沒有 GPU 的環境也能跑)
    try:
        state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
        model.load_state_dict(state_dict)
        model.eval()
        return model
    except Exception as e:
        st.error(f"❌ 載入權重錯誤，可能檔案損毀: {e}")
        return None

# 初始化載入模型
model = load_princess_model()

# ==========================================
# 🖼️ 3. 影像預處理設定
# ==========================================
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 📤 4. 前端 UI 與預測邏輯
# ==========================================
uploaded_file = st.file_uploader("選擇一張公主圖片...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # 顯示上傳的圖片
    image = Image.open(uploaded_file).convert("RGB")
    st.image(image, caption="📷 上傳的圖片", use_container_width=True)
    
    st.write("🧠 AI 辨識中...")
    
    if model is not None:
        # 影像預處理
        input_tensor = transform_pipeline(image).unsqueeze(0) # 增加 Batch 維度
        
        # 進行預測
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)
        
        # 顯示預測結果
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        score = confidence.item() * 100
        
        st.success(f"🎉 辨識結果：**{predicted_class}**")
        st.info(f"📊 信心度 (Confidence)：**{score:.2f}%**")
        
        # 顯示前三名機率
        st.write("---")
        st.write("🔍 **前 3 名候選機率排名：**")
        top3_prob, top3_idx = torch.top_k(probabilities, 3)
        for i in range(3):
            st.write(f"{i+1}. {CLASS_NAMES[top3_idx[i].item()]}: **{top3_prob[i].item()*100:.2f}%**")
