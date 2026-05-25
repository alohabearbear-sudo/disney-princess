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

st.title("👑 AI的精準度：玻璃鞋能幫王子找對公主嗎？")
st.write("上傳一張迪士尼公主的圖片，讓玻璃鞋幫忙辨識她是誰！")

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
# ⚙️ 2. 載入模型權重 (修正快取 UI 衝突問題)
# ==========================================
# 保持這個函數乾淨，裡面絕對不放任何 st.xxx 元件
@st.cache_resource
def load_princess_model_core(weights_path, download_url, num_classes):
    # 建立與訓練時完全相同的 ResNet-50 架構
    model = models.resnet50(weights=None)
    model.fc = nn.Linear(model.fc.in_features, num_classes)
    
    # 檢查本地是否存在權重，若無則聯網下載
    if not os.path.exists(weights_path):
        urllib.request.urlretrieve(download_url, weights_path)
                
    # 載入權重 (強制對應到 CPU)
    state_dict = torch.load(weights_path, map_location=torch.device('cpu'))
    model.load_state_dict(state_dict)
    model.eval()
    return model

# 這個包裝函數放在外面，負責處理 UI 動畫與錯誤提示
def get_model():
    weights_path = "resnet50_fold_1_best.pth"
    download_url = "https://github.com/alohabearbear-sudo/disney-princess/releases/download/v1/resnet50_fold_1_best.pth"
    num_classes = len(CLASS_NAMES)
    
    try:
        # 如果檔案不存在，我們在快取函數外面秀出下載轉圈圈
        if not os.path.exists(weights_path):
            with st.spinner("📥 正在從 GitHub 下載雲端模型權重（僅需下載一次，請稍候...）"):
                model = load_princess_model_core(weights_path, download_url, num_classes)
            st.toast("🎉 模型權重下載成功！", icon="✅")
        else:
            # 如果檔案早就存在，直接靜悄悄地載入快取
            model = load_princess_model_core(weights_path, download_url, num_classes)
        return model
    except Exception as e:
        st.error(f"❌ 載入模型失敗！原因: {e}")
        return None

# 🚀 正式初始化載入模型
model = get_model()

# ==========================================
# 🖼️ 3. 影像預處理設定
# ==========================================
transform_pipeline = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225])
])

# ==========================================
# 📤 4. 前端 UI 與預測邏輯 (HTML 語法外放修正版)
# ==========================================
uploaded_file = st.file_uploader("選擇一張公主圖片...", type=["jpg", "jpeg", "png", "bmp"])

if uploaded_file is not None:
    # 讀取並轉換圖片
    image = Image.open(uploaded_file).convert("RGB")
    
    # 先跑 AI 預測邏輯
    if model is not None:
        input_tensor = transform_pipeline(image).unsqueeze(0)
        with torch.no_grad():
            outputs = model(input_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
            confidence, predicted_idx = torch.max(probabilities, 0)
        
        predicted_class = CLASS_NAMES[predicted_idx.item()]
        score = confidence.item() * 100

    # 使用 st.columns 切成左右兩邊
    col1, col2 = st.columns(2)
    
    # ---- 👈 左側欄位：放上傳的圖片與主要預測結果 ----
    with col1:
        st.image(image, caption="📷 上傳的圖片", use_container_width=True)
        st.write("🧠 AI 辨識中...")
        st.success(f"🎉 辨識結果：**{predicted_class}**")
        st.info(f"📊 信心度 (Confidence)：**{score:.2f}%**")
        
    # ---- 👉 右側欄位：自製緊湊型進度條 (修正 Markdown 位置) ----
    with col2:
        st.write("🔍 **所有 11 名公主候選機率排名：**")
        
        # 進行 11 名的大到小排序
        all_prob, all_idx = torch.topk(probabilities, len(CLASS_NAMES))
        
        # 1. 在迴圈外面，初始化 HTML 容器外殼
        html_content = '<div style="margin-top: 0px;">'
        
        # 2. 進入迴圈，這裏「只做字串拼接」，絕對不呼叫任何 st.xxx
        for i in range(len(CLASS_NAMES)):
            prob_value = all_prob[i].item()      
            prob_percentage = prob_value * 100   
            class_name = CLASS_NAMES[all_idx[i].item()]
            
            # 每個進度條的高度與間距經過精密微調 (margin-bottom: 3px)，確保底部貼齊左邊
            html_content += f"""
            <div style="margin-bottom: 4px; font-size: 14px; line-height: 1.1; font-family: sans-serif;">
                <div style="display: flex; justify-content: space-between; margin-bottom: 2px;">
                    <span>{i+1}. {class_name}</span>
                    <strong>{prob_percentage:.2f}%</strong>
                </div>
                <div style="background-color: #f0f2f6; border-radius: 4px; height: 6px; width: 100%;">
                    <div style="background-color: #2b5c8f; height: 6px; border-radius: 4px; width: {prob_percentage}%;"></div>
                </div>
            </div>
            """
            
        # 3. 離開迴圈後，把 HTML 容器封口
        html_content += '</div>'
        
        # 4. 🔥 關鍵修正：在迴圈最外面，只呼叫這一次 st.markdown 渲染全部 11 個進度條！
        st.markdown(html_content, unsafe_allow_html=True)
