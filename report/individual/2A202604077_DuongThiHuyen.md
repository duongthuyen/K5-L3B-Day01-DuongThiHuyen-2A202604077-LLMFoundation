# Báo cáo cá nhân — Day 01: Lọc tin nhắn rác bằng ML, DL, Embedding và LLM

- **MSSV:** 2A202604077
- **Họ và tên:** Dương Thị Huyền
- **SEED** (in ở ô đầu notebook): 9517

## 1. Bằng chứng

| Tệp | Có? |
| --- | --- |
| [`day01_lab.ipynb`](day01_lab.ipynb) — notebook đã chạy, còn output | ☑ |
| [`lab1_report.json`](lab1_report.json) — do ô "Báo cáo cuối" sinh ra | ☑ |
| [`hand_calc.jpg`](hand_calc.jpg) — ảnh tính tay có các bước | ☑ |

Ô báo cáo cuối đạt 15/15 bộ kiểm tra (10 ô code và 5 giá trị tính tay), đủ 10 dự đoán và mỗi dự đoán đều được ghi trước thí nghiệm tương ứng. Giờ trong log là giờ UTC của máy Colab, chậm 7 tiếng so với giờ Việt Nam.

## 2. Kết quả chính

| Cách tiếp cận | F1 test | F1 SUB 40 | p95 ms/tin | USD/1k tin |
| --- | --- | --- | --- | --- |
| 1. BoW + LogReg (tự code) | 0.861 | 0.824 | 0.09 | 0 |
| 2. BoW + MLP | 0.927 | 0.889 | 0.36 | 0 |
| 3a. Encoder (MiniLM) + kNN | 0.924 | 0.974 | 84.08 | 0 |
| 3b. Encoder (MiniLM) + LogReg | 0.951 | 0.923 | 33.09 | 0 |
| 3c. Encoder (MiniLM) + MLP head | 0.951 | 0.947 | 8.08 | 0 |
| 4a. LLM nhỏ zero-shot (gpt-6-luna) | | 0.000 | 1637.30 | 0.006 |
| 4b. LLM lớn zero-shot (gpt-6.1-terra) | | 0.000 | 1120.11 | 0.056 |
| 4c. LLM nhỏ few-shot | | 0.000 | 1010.58 | 0.021 |

Các hàng LLM chỉ chạy trên SUB nên không có F1 test. Cả 3 hàng LLM đều có invalid = 40/40: model được gọi thành công (độ trễ khoảng 1 giây mỗi tin) nhưng câu trả lời đều rỗng, nên mọi tin bị tính là ham và F1 bằng 0. Chi phí ghi nhận đúng bằng phần token input, tức là model không trả về token nào, trong khi ô thử với max_tokens=20 vẫn trả lời bình thường. Nguyên nhân là giới hạn 3 token mặc định trong `PROVIDERS` của đề quá nhỏ với model này. Mình giữ nguyên cấu hình của đề và ghi nhận kết quả như trên. Vì vậy chưa so sánh được chất lượng của LLM với các cách khác, chỉ so được độ trễ và chi phí: model lớn đắt hơn model nhỏ khoảng 10 lần, few-shot đắt hơn zero-shot khoảng 3.5 lần vì prompt dài hơn.

Trong các cách còn lại, 3b và 3c có F1 test cao nhất (0.951). Độ trễ thay đổi giữa các lần chạy vì phụ thuộc máy Colab, các số trên là của lần chạy cuối.

## 3. Dự đoán và thực tế

**Learning rate (ml_lr).** Mình đoán lr=50 bước nhảy quá lớn nên loss sẽ dao động, khó giảm đều. Thực tế loss vọt lên khoảng 2.6 ở vài bước đầu rồi giảm rất nhanh, kết thúc ở 0.006, còn thấp hơn lr=0.5 (0.083). Mình lệch vì quên rằng loss của logistic regression là hàm lồi, gradient bị chặn (error nằm trong khoảng -1 đến 1, feature là 0 hoặc 1) và dữ liệu train gần như tách được. Bước lớn chỉ gây dao động lúc đầu, sau đó đẩy trọng số lên rất lớn để khớp gần hết tập train. Loss train thấp như vậy cũng chưa chắc tốt hơn trên test.

**Tin nhắn ngụy trang (emb_tricky).** Mình đoán Encoder + LogReg đúng khoảng 5/6, BoW khoảng 3 hoặc 4/6. Thực tế ngược lại: BoW đúng 5/6, Encoder + LogReg đúng 4/6. Tin "Your mobile number has been chosen for a complimentary holiday. Reply YES to confirm" vẫn chứa nhiều từ có trọng số spam cao trong vocab như your, reply, mobile nên BoW bắt được, còn với embedding thì nghĩa cả câu khá giống một tin xác nhận bình thường nên rơi vào vùng ham. Mình đã đánh giá thấp lượng từ khoá spam còn sót trong các tin này. Với chỉ 6 tin thì chênh 1 tin cũng chưa nói lên nhiều.

**Bỏ ReLU (dl_relu).** Mình đoán bỏ ReLU thì mạng tương đương logistic regression nên F1 sẽ giảm về gần 0.861 của Phần 1. Thực tế F1 vẫn là 0.927, bằng MLP có ReLU. Ý tương đương mô hình tuyến tính là đúng, nhưng mình lệch ở chỗ cho rằng MLP hơn Phần 1 là nhờ phi tuyến. Thật ra logistic regression ở Phần 1 mới train 300 epoch với gradient descent thường (F1 train chỉ 0.912), còn mạng ở Phần 2 được train bằng Adam theo mini-batch 30 epoch nên khớp dữ liệu tốt hơn. Phần chênh lệch đến từ cách train chứ không phải từ ReLU.

## 4. Phân tích 50 case khó

BoW + LogReg tụt nhiều nhất: F1 từ 0.824 trên SUB xuống 0.333 trên 50 case khó, accuracy chỉ còn 0.52. BoW + MLP cũng giảm mạnh (0.889 xuống 0.474). Các model embedding giữ được tốt hơn, cao nhất là kNN với F1 0.737. Với BoW + LogReg, các nhóm khó nhất là Spam giọng tự nhiên (đúng 10%), Thao túng / lừa đảo (12%) và Né từ khoá (17%). Ngược lại BoW đúng 100% nhóm Tưởng spam nhưng không phải, vì mô hình rất dè dặt (precision 1.0 trên test) nên vài từ khoá trong một tin ngắn chưa đủ vượt ngưỡng 0.5. Có 5 case mà tất cả model đều sai, đều thuộc nhóm Spam giọng tự nhiên.

**Case #29 (Né từ khoá):** "C0ngr4tulat10ns! Y0u h4ve w0n a £1000 g1ft c4rd. Cl4im n0w at w1n-gift.c0m". BoW chỉ cho p = 0.074. Khi thay chữ bằng số, các từ như c0ngr4tulat10ns, w0n, cl4im trở thành token lạ không có trong vocab nên bị bỏ qua hoàn toàn. Những token còn lại chủ yếu là từ thông dụng, không đủ điểm spam. Embedding làm tốt hơn hẳn ở nhóm này (kNN và 3c đúng 83%) vì encoder chia từ thành các mảnh nhỏ hơn và vẫn nhận ra được ý trúng thưởng.

**Case #21 (Thao túng / lừa đảo):** "Hi Mum, I dropped my phone in the toilet, this is my new number. Can you send £200 to my friend's account?". BoW chỉ cho p = 0.021, tức là rất chắc chắn là ham. Tin này không có từ khoá spam nào, lại toàn các từ mà mô hình học được là kéo về phía ham như i, my, me. BoW chỉ cộng trọng số từng từ nên không hiểu được ý định của tin là giả làm người thân để xin chuyển tiền.

Tập test thường quá lạc quan vì nó cùng phân phối với tập train, nhiều spam là các mẫu gần giống nhau chỉ khác số điện thoại hay mã thưởng. 50 case khó được viết để tránh đúng những mẫu đó nên mới lộ ra là model chưa thật sự hiểu thế nào là spam.

## 5. Thử thách cá nhân

**Đề #2:** giảm vocab còn 100 từ và tăng lên 5000 từ, train lại, báo F1 test và giải thích xu hướng.

**Cách làm:** với mỗi kích thước 100, 2000 và 5000, mình xây lại vocab bằng `build_vocab(texts_tr, max_size=...)`, vector hoá lại tập train và test, train lại bằng `train_logreg` với cùng `lr=0.5, epochs=300` như Phần 1 để so sánh công bằng, rồi tính metrics trên cả train và test. Mình dùng tên biến riêng để không ghi đè `vocab`, `w`, `b` của Phần 1.

| Vocab | F1 train | Precision test | Recall test | F1 test |
| --- | --- | --- | --- | --- |
| 100 | 0.837 | 0.952 | 0.638 | 0.764 |
| 2000 | 0.912 | 1.000 | 0.755 | 0.861 |
| 5000 | 0.917 | 1.000 | 0.755 | 0.861 |

**Giải thích:** giảm còn 100 từ thì F1 test tụt xuống 0.764, chủ yếu do recall giảm. 100 từ phổ biến nhất phần lớn là từ chức năng như i, to, you, the, chỉ còn vài từ spam như call, free, txt, text, reply, stop, nên nhiều tin spam không còn đủ tín hiệu và bị lọt. Tăng lên 5000 từ thì F1 test không đổi, chỉ F1 train nhích lên 0.917. Từ hạng 2000 chỉ xuất hiện 3 lần trong tập train, từ hạng 5000 chỉ 1 lần. Vì `grad_w = X.T @ error / n`, trọng số của từ hiếm gần như không được cập nhật và vẫn gần 0, nên dù 321/775 tin test có chứa từ mới thì dự đoán vẫn không đổi. Xu hướng là tăng vocab giúp nhiều lúc đầu rồi bão hoà, khoảng 2000 từ là hợp lý cho dữ liệu này.

## 6. Phương án cho sản phẩm thật

Với 10 triệu SMS/ngày (khoảng 116 tin/s trung bình) và yêu cầu p95 < 50 ms, mình chọn **3c. Encoder (MiniLM) + MLP head** làm model chính. Đây là cách có F1 test cao nhất (0.951, ngang 3b) trong khi p95 chỉ 8.1 ms, và xử lý theo lô được khoảng 1.500 tin/s, dư cho mức trung bình. 3b cũng đạt SLA nhưng chậm hơn (33.1 ms). kNN có F1 trên SUB cao nhất (0.974) nhưng p95 84 ms, vượt SLA, và theo lô chỉ khoảng 55 tin/s vì phải so mỗi tin với toàn bộ tập train. LLM mất khoảng 1 giây mỗi tin. Nếu dùng LLM nhỏ cho toàn bộ thì khoảng 10.5 USD cho 1 triệu tin, tức khoảng 105 USD/ngày, tầm 3.150 USD/tháng, mà vẫn không đạt SLA.

Để tận dụng thêm các cách khác, mình đề xuất một router: BoW + LogReg chạy trước cho mọi tin vì gần như không tốn gì (p95 0.09 ms), tin có p dưới 0.2 hoặc trên 0.8 thì quyết định luôn. Tin lưỡng lự chuyển sang 3c. Chỉ những tin vẫn lưỡng lự sau 3c, hoặc bị người dùng báo cáo, mới gửi LLM xử lý bất đồng bộ, không nằm trên đường gửi tin.

Đánh đổi chấp nhận được: 3c vẫn sai nhiều ở spam giọng tự nhiên và tin nói về spam (theo 50 case khó), cần thêm phần cứng cho encoder và phải cập nhật dữ liệu có nhãn định kỳ khi có kiểu spam mới.

## 7. AI log và điều đã học

| Phần | Hỏi AI gì | Nhận được gì | Mình đã sửa / kiểm chứng gì |
| --- | --- | --- | --- |
| Phần 1 đến 3 | Nhờ review code TODO đã viết | Góp ý những chỗ chưa đúng yêu cầu | Sửa lại, chạy các ô kiểm tra |
| Phần 4 | Hỗ trợ code LLM và tìm nguyên nhân F1 = 0 | Chỉ ra lỗi đọc nhãn và giới hạn 3 token | Tự tính lại chi phí để kiểm chứng |
| Các câu hỏi và báo cáo | Hỗ trợ phân tích output và diễn đạt câu trả lời | Gợi ý cách giải thích dựa trên số liệu | Đối chiếu với output của mình, chỉnh theo cách hiểu |
| Thử thách #2 | Hỗ trợ code và giải thích kết quả | Vòng lặp train với 3 kích thước vocab | Chạy code, đối chiếu kết quả |

Điều mình hiểu ra hôm nay: điểm cao trên tập test chưa chắc là model hiểu bài toán. Cùng một model có thể đạt F1 0.86 trên tập test nhưng chỉ 0.33 trên 50 case khó, nên cần thử trên dữ liệu khác phân phối. Mình cũng thấy trực giác dễ sai, như lr lớn vẫn hội tụ hay ReLU không phải lý do MLP tốt hơn. Cuối cùng là một lỗi nhỏ ở bước gọi API hoặc đọc kết quả có thể làm cả phần LLM ra F1 bằng 0, nên phải luôn xem output thô và số câu trả lời invalid trước khi kết luận về model.
