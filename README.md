# BKAI_Challenge_Vietnamese_OCR
BKAI-NAVER 2023 - Track 3: OCR

Giới thiệu chung
Chủ đề. Cuộc thi Vietnamese Handwritten Text Recognition tập trung vào giải quyết bài toán "Nhận dạng văn bản chữ viết tay Tiếng Việt ".

Nhiệm vụ. Cuộc thi tập trung duy nhất vào một nhiệm vụ: nhận dạng chữ viết tay tiếng Việt.

Dữ liệu

Dữ liệu được cung cấp bởi ban tổ chức gồm 3 tập như sau:
Training data: là tập dữ liệu thật có gán nhãn, dùng để huấn luyện mô hình. Tập này gồm 103000 ảnh (Gồm 51000 ảnh form, 48000 ảnh wild và 4000 ảnh GAN).
Public test: Là tập dữ liệu không nhãn sử dụng để đánh giá vòng sơ loại. Tập này gồm 33000 ảnh. (Gồm 17000 ảnh form và 16000 ảnh wild)
Private test: Là tập dữ liệu không có nhãn. Thông tin chi tiết sẽ công bố tại Vòng chung kết.
Đầu vào cho mô hình là các ảnh thô chưa được gán nhãn. Tệp nhãn là các file định dạng .txt. Mỗi dòng của tệp nhãn chứa thông tin là tên ảnh và nhãn của văn bản chứa trong ảnh đó theo khuôn dạng như sau:
                    IMAGE_NAME   GROUND_TRUTH_TEXT 

Tiêu chí đánh giá. Tiêu chí đánh giá là chỉ số CER, đại diện cho phần trăm ký tự trong văn bản của tệp nhãn bị dự đoán không chính xác. CER càng thấp thì mô hình nhận diện càng chính xác (Chi tiết xem tại phần Evaluation).

Trang web chính thức cuộc thi: https://bkai.ai/soict-hackathon-2023/
