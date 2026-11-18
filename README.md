# RSSViz：robosimstudio 分身的物理記錄觀測

RSSViz 用來看 [robosimstudio](https://github.com/robosimstudio/robosimstudio) live 分身跑完 routine 之後的物理資料，包括誰碰到誰、力有多大、接觸點在哪裡、物體怎麼動，以及關節有沒有跟上目標。

它可以回答這類問題：
- 手指夾住物體的力有多大？
- 物體被放下時，接觸點落在哪裡？
- 有沒有不該碰的連桿碰到了東西？

## 從 ExpViz 改來的

RSSViz 參考 [**ExpViz**](https://github.com/charles33502/ExpViz) 改寫而來。ExpViz 是通用的實驗數據觀測工具，能把任何 JSON、JSONL、CSV、NPZ 拆成表格再畫圖。

**沿用 ExpViz 的部分：**
- 技術架構：Streamlit 網頁加 Plotly 圖表，跑在同一個 conda 環境 `expviz`（streamlit、plotly、pandas、numpy）。
- 啟動腳本 `run.sh` 的寫法：可帶資料夾和 port 兩個參數，只聽 localhost。
- 每張圖下方都能展開「這張圖的數據」並下載 CSV。
- 用 Streamlit `AppTest` 做無頭測試（`test_app.py`）。

**另外做 RSSViz 的原因：** ExpViz 不認得 robosimstudio 記錄的結構，只把每個接觸點當成表格裡的一列。直接畫出來的結果，例如把每一步的所有 `normal_force` 取平均，會把桌子壓地板、手指夾物體、物體壓桌子等毫不相干的力混在一起，沒有物理意義。

**RSSViz 改變的地方：**

| | ExpViz | RSSViz |
|---|---|---|
| 讀什麼 | 任何表格狀資料 | 分身的 `run/<name>_phys.npz`、`_contacts.jsonl`、`_shapes.npz` |
| 接觸的單位 | 一列一個點 | 換算到剛體，再分組成左手指、右手指、左臂、右臂和各個物體 |
| 力 | 原始欄位 | 「A 受 B 的力」，正負號自動處理，同一步的點先加總 |
| 3D | 無 | 瀏覽器端的 Three.js 檢視器：場景形狀、接觸點、法向箭頭、時間軸播放 |
| 驗證 | 程式不出錯 | 另外檢查物體靜置時的支撐力是否等於重量（`test_rssdata.py`） |

ExpViz 本身沒有改動，兩者可以同時使用：ExpViz 預設在 port 8501，RSSViz 在 8502。

## 資料從哪裡來

分身資料夾裡的 `physlog.py` 會包住分身的 routine，在 live 頁面上變成 **routine (logged)** 按鈕，也可以用 `./ctl play "routine (logged)" --wait` 執行。跑完後在分身的 `run/` 寫出三個檔：

| 檔案 | 內容 |
|---|---|
| `routine_phys.npz` | 每個控制步（50 Hz）的剛體位姿 `body_q`、速度 `body_qd`，關節的實際角度 `joint_q`、角速度 `joint_qd`、目標角度 `joint_target_q`，以及名稱對照表 |
| `routine_contacts.jsonl` | 每步一行，列出那一步的每一個接觸點：兩個形狀、作用在第二個形狀上的力（N）、位置、法向、法向力和摩擦力，以及穿入深度 |
| `routine_shapes.npz` | 每個形狀的類型、尺寸、相對所屬剛體的位置、網格頂點，只寫一次，用來畫出場景 |

`physlog.py` 讀的是 MuJoCo-Warp 的接觸資料：力的部分經過 robosimstudio 的 `ContactProbe`，位置和法向則直接讀 `data.contact` 的 `pos` 和 `frame`。robosimstudio 函式庫本身沒有改。目前有 `physlog.py` 的分身是 `v22139905` 和 `v22070938`。

## 安裝與啟動

`run.sh` 用的是 conda 環境 `expviz`（和 [ExpViz](https://github.com/charles33502/ExpViz) 共用）。從頭建立：

```bash
conda create -n expviz python=3.12
conda run -n expviz pip install -r requirements.txt
```

啟動：

```bash
~/rssviz/run.sh                      # 預設讀 ~/robosimstudio/examples/live/twins，port 8502
~/rssviz/run.sh <twins 資料夾> 8503  # 指定資料夾和 port
```

啟動後打開 http://localhost:8502。
要讓同一個網段的其他電腦連進來，啟動前先設 `RSSVIZ_ADDR=0.0.0.0`。

## 分頁

| 分頁 | 內容 |
|---|---|
| 🤝 接觸力 | 選受力方 A 和施力方 B，畫出 B 對 A 的法向力總和、摩擦力、合力和 Fx、Fy、Fz，以及接觸點數。可以加一條重量線 m·g，也有「曾經接觸過的組合」清單可一鍵帶入 |
| 📍 接觸點分布 | 3D 檢視器（見下節） |
| 📦 物體運動 | 位移、高度變化、速度、角速度、相對起始姿態的轉角，並附一張摘要表 |
| 🦾 關節 | 左臂、右臂各關節的實際角度對目標角度，或兩者的差（追蹤誤差） |
| ✅ 物理檢查與錄影 | 分身 `run/physics.json` 的各項檢查（`./ctl physics`），以及 take 影片 |
| ℹ️ 說明 | 力的定義和注意事項 |

### 3D 檢視器
- **場景形狀**：
  - 碰撞形狀（預設）：物理引擎實際用來算接觸的形狀。
  - 外觀網格：機器人的外觀模型，比較細，資料約 11–12 MB，載入較慢。
  - 不畫。
- **接觸點**：球的顏色和大小代表單點法向力。色階上限固定為整段記錄單點法向力的第 99 百分位，所以不同時間的顏色可以直接比較。紅色箭頭是法向乘以力。
- **時間軸、播放、逐步前進後退都在瀏覽器裡處理**，切換時間不會重設視角。視角也會記在瀏覽器裡，改上方選項重新載入後會沿用。
- **鏡頭**：
  - 右上角面板：平移、縮放、繞轉、俯仰，以及正面、側面、俯視、斜角、全場景五種預設視角。
  - 滑鼠：左鍵拖曳旋轉，右鍵或 Shift + 拖曳平移，滾輪縮放。
  - 鍵盤：方向鍵平移，W/S 縮放，Q/E 繞轉，R/F 俯仰，C 對準接觸點，空白鍵播放，`,` `.` 逐步，V 錄影。
- **錄影（⏺ 錄影 或 V）**：從目前這一格到最後，逐格輸出成 MP4，完成後自動下載。
  - 依目前的視角與播放速度輸出（0.25× 就是 4 倍長的慢動作），固定 30 fps，最大 1920×1080。
  - 畫面上會疊上這一刻各組接觸的法向力總和、色條和時間。
  - 輸出中按鈕顯示進度，再按一次提早結束，已輸出的部分照樣存檔。
  - 用瀏覽器的 WebCodecs 編碼 H.264，再用 [mp4-muxer](https://github.com/Vanilagy/mp4-muxer) 5.1.3 封裝成一般 MP4（`moov` 在檔頭、不分段），可以直接插入 PowerPoint。需要 Chrome / Edge 94 以上。
  - 不支援 WebCodecs 的瀏覽器會改用 `MediaRecorder` 即時錄影。那種檔案是分段式 MP4 或 WebM，檔頭長度為 0，PowerPoint 不收，要先用 ffmpeg 轉檔：
    ```bash
    ffmpeg -i in.mp4 -c:v libx264 -pix_fmt yuv420p -r 30 -movflags +faststart out.mp4
    ```
- **左側圖例**：可勾選顯示或隱藏每個物體和每組接觸，並顯示這一刻每組接觸的法向力總和。
- 滑鼠移到接觸點上，會顯示兩個形狀的名稱、法向力、摩擦力和穿入深度。

## 力的定義與注意事項

- 每個接觸點的 `force` 是作用在第二個形狀 `b` 上的力，單位 N，世界座標，z 朝上。它等於 `ContactProbe` 的 impulse 除以物理子步長 0.002 s。
- **法向力**是沿接觸法向、兩者互相壓的分量，不管 a、b 的順序都是正值。**摩擦力**是扣掉法向後的切向分量；同一步裡會先把各點的切向力向量相加，再取大小。
- **請看加總，不要看單點。** 同一對物體在同一步通常有好幾個接觸點，MuJoCo 怎麼把力分到各點並不唯一，而且每步會跳動。單點的數值適合用來看「哪裡有碰到」。
- 記錄的是每個控制步（0.02 s）**最後一個**物理子步（0.002 s）的值，不是 10 個子步的平均。碰撞瞬間的尖峰可能被漏掉或被放大。
- `depth` 是 MuJoCo 的有號距離，負值代表穿入。
- 3D 檢視器要從 cdn.jsdelivr.net 載入 three.js 0.160.0（錄影時另外載入 mp4-muxer 5.1.3），所以瀏覽器必須能連網。

## 測試

```bash
cd ~/rssviz
~/miniconda3/envs/expviz/bin/python test_rssdata.py   # 物理檢查：靜置時其他東西對物體的 Fz = 物體重量
~/miniconda3/envs/expviz/bin/python test_app.py       # 每個分身的每個分頁跑一次，確認沒有例外
```

`test_rssdata.py` 裡物體的質量是照各分身 `scene.py` 寫的；加新分身時要一併補上。上次執行的結果：

| 物體 | 支撐力 | 重量 |
|---|---|---|
| 槌子 | 4.41 N | 4.41 N |
| 收納盒 | 1.18 N | 1.18 N |
| 菜刀 | 3.43 N | 3.43 N |
| 砧板 | 8.83 N | 8.83 N |

## 檔案

| 檔案 | 用途 |
|---|---|
| `app.py` | Streamlit 網頁：側欄選分身和記錄，6 個分頁 |
| `rssdata.py` | 讀檔、剛體分組、力的計算、形狀網格、3D 檢視器的資料 |
| `viewer.html` | Three.js 檢視器的範本，由 `app.py` 填入資料；錄影也在這裡 |
| `physlog.py` | 放進分身資料夾的記錄器，寫出上面三個檔 |
| `run.sh` | 啟動腳本 |
| `requirements.txt` | Python 套件 |
| `test_rssdata.py`、`test_app.py` | 測試 |

## 給新分身用

1. 把這個 repo 的 `physlog.py` 複製到新分身的資料夾（和 `scene.py` 同一層）。
2. 在 `routine.py` 最後加上：
   ```python
   from physlog import logged
   routines["routine (logged)"] = logged(<routine 函式>, "routine")
   ```
3. 在 live 頁面按 **routine (logged)**。
4. 在 RSSViz 側欄按「重新讀取」，新分身就會出現在選單裡。

## 授權

MIT，見 [LICENSE](LICENSE)。
