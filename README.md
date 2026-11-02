# NOISE / LAB — IMP302 mini project

Mini project xử lý ảnh cho IMP302, Đại học FPT TP. Hồ Chí Minh. Bài toán là khử nhiễu Gaussian trên ảnh tự nhiên mức xám. Phần mềm tạo ảnh nhiễu, chạy Gaussian, bilateral, Non-Local Means (NLM), một phép hợp nhất có trọng số học từ dữ liệu và một MLP học phần dư. Kết quả được đo bằng PSNR/SSIM trên ảnh giữ riêng và trình bày trong demo web cục bộ.

## Bắt đầu

Yêu cầu: Python 3.10+, pip và mạng để lấy dữ liệu từ Berkeley lần đầu. Trên Windows PowerShell, chạy trong thư mục `mini-project`:

```powershell
python -m venv .venv
.\.venv\Scripts\python.exe -m pip install -r requirements.txt
.\.venv\Scripts\python.exe -m src.experiment
.\.venv\Scripts\python.exe report\make_results.py
.\.venv\Scripts\python.exe -m src.verify
.\.venv\Scripts\python.exe demo\server.py
```

Mở `http://127.0.0.1:8765`. Lần chạy thí nghiệm tạo `data/manifest.json`, ảnh xử lý trong `data/clean`, mô hình trong `models` và bảng/đồ thị trong `results`. Tệp tải gốc đã có tại `data/raw/BSDS300-images.tgz`; mã sẽ xác thực SHA-256. Tự động tải lại nếu tệp vắng mặt.

PDF báo cáo **tiếng Anh**, ghi hai thành viên Huỳnh Quốc Việt (SE194225), Lê Nguyễn Gia Hưng (SE194127) và giảng viên NguyenLP9, là [`report/IMP302_Mini_Project.pdf`](report/IMP302_Mini_Project.pdf). Nguồn là [`report/main.tex`](report/main.tex). Biên dịch lại bằng XeLaTeX sau khi tạo `report/results.tex` và chạy hai notebook EDA:

```powershell
cd report
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
xelatex -interaction=nonstopmode -halt-on-error -output-directory=build main.tex
```

Gói sẵn để import vào Overleaf: [`report/IMP302_Overleaf.zip`](report/IMP302_Overleaf.zip); xem [`report/OVERLEAF.md`](report/OVERLEAF.md) để chọn XeLaTeX. Để học nội dung báo cáo bằng tiếng Việt đơn giản, đọc [`docs/teach-me-like-grade-5.md`](docs/teach-me-like-grade-5.md).

## EDA và notebook

Hai notebook đã chạy và lưu hình/đầu ra: [`01_dataset_eda.ipynb`](notebooks/01_dataset_eda.ipynb) kiểm tra nguồn, split và ảnh train; [`02_features_and_model.ipynb`](notebooks/02_features_and_model.ipynb) trực quan hóa bộ lọc, mười feature, residual MLP và quá trình chọn mô hình. Nhận xét có dẫn tới số liệu trong [`docs/eda-findings.md`](docs/eda-findings.md). Chạy lại từ thư mục `mini-project`:

```powershell
.\.venv\Scripts\python.exe -m pip install -r requirements-notebooks.txt
.\.venv\Scripts\python.exe -m ipykernel install --prefix .venv --name imp302 --display-name "IMP302 (.venv)"
.\.venv\Scripts\python.exe notebooks\run_notebooks.py
.\.venv\Scripts\python.exe report\make_results.py
.\.venv\Scripts\python.exe report\package_overleaf.py
```

EDA khám phá pixel **chỉ trên train**. Notebook có kiểm tra metadata và khả năng trùng ảnh giữa các split; kết quả validation/test đã lưu được đọc để giải thích, không dùng để đổi mô hình.

## Cấu trúc

```text
mini-project/
├── data/raw/            # archive Berkeley gốc, URL/SHA trong src/data.py
├── data/clean/          # ảnh mức xám đã resize theo split
├── data/manifest.json   # ID train, val, pilot và test cuối
├── src/                 # dữ liệu, lọc, mô hình, thí nghiệm
├── models/              # trọng số MLP và trọng số fusion
├── results/             # CSV theo ảnh, tóm tắt, bootstrap, hình
├── notebooks/           # hai notebook EDA đã chạy và hình được lưu
├── demo/                # HTTP server cục bộ và giao diện
├── report/              # nguồn LaTeX tiếng Anh, PDF, gói Overleaf
└── docs/                 # đối chiếu môn học, bài giảng dễ hiểu, ghi nhận EDA
```

## Thiết kế nghiên cứu

- Dữ liệu: ảnh BSDS300 chính thức; chỉ dùng cho nghiên cứu/giáo dục phi thương mại. 60 ảnh train và 12 ảnh validation thuộc split train gốc. 20 ảnh `pilot` từ test gốc được dùng trong quá trình phát triển; 20 ảnh test tiếp theo không tham gia quyết định thiết kế và là nguồn số liệu cuối.
- Nhiễu: Gaussian tổng hợp với σ = 15, 25, 50; seed 302 dẫn xuất theo ID và mức nhiễu. Ảnh sạch tham chiếu là JPEG Berkeley đã giải mã, chuyển mức xám và resize cạnh dài ≤256 pixel.
- NLM được chọn h theo PSNR validation. Trọng số fusion không âm, tổng bằng 1, được fit từ train cho từng σ. MLP 1 tầng ẩn được huấn luyện từ train; số nút và đặc trưng texture chọn theo validation.
- Tệp `results/per_image.csv` chứa điểm từng ảnh/phương pháp. `results/paired_bootstrap.csv` có chênh lệch theo cặp ảnh và khoảng tin cậy 95%. Không suy diễn các khoảng này cho nhiễu thực hoặc mọi ảnh tự nhiên.
- Cần xem cả PSNR, SSIM, kết quả trực quan và baseline mạnh nhất. Một biến thể học máy không mặc nhiên tốt hơn mọi bộ lọc.

## Nguồn và giới hạn

Nguồn dữ liệu: [Berkeley Segmentation Dataset](https://www2.eecs.berkeley.edu/Research/Projects/CS/vision/bsds/). NLM dựa trên Buades et al. (CVPR 2005), bilateral trên Tomasi & Manduchi (ICCV 1998), SSIM trên Wang et al. (TIP 2004). Danh mục nguồn đầy đủ trong `report/main.tex`. Đây là bài thực nghiệm trên ảnh mức xám và nhiễu mô phỏng; chưa xác nhận cho ảnh màu hoặc ảnh nhiễu camera thực.
