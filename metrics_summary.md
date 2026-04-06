# Tổng hợp số liệu mô hình ESG-Washing

Dưới đây là phần tóm tắt kết quả huấn luyện (Loss, Macro F1, Micro F1) so sánh giữa hai mô hình: **Baseline** (không sử dụng Neuro-symbolic) và **Neuro-Symbolic** trên 2 tác vụ: _Topic Classification_ và _Actionability Classification_, được trích xuất từ notebook `train-model.ipynb`.

---

## 1. Tác vụ Topic Classification (Phân loại Chủ đề)

### Mô hình Baseline

| Epoch | Training Loss | Validation Loss | Macro F1 | Micro F1 |
| ----- | ------------- | --------------- | -------- | -------- |
| 1     | 0.445859      | 0.397618        | 0.774177 | 0.864725 |
| 2     | 0.333870      | 0.371381        | 0.810666 | 0.893850 |
| 3     | 0.274824      | 0.370518        | 0.821114 | 0.902360 |
| 4     | 0.240731      | 0.379678        | 0.822643 | 0.905693 |
| 5     | 0.214089      | 0.420639        | 0.830846 | 0.909290 |

### Mô hình Neuro-Symbolic

| Epoch | Training Loss | Validation Loss | Macro F1 | Micro F1 |
| ----- | ------------- | --------------- | -------- | -------- |
| 1     | 0.604175      | 0.557609        | 0.763519 | 0.852180 |
| 2     | 0.459228      | 0.514513        | 0.803492 | 0.892008 |
| 3     | 0.420371      | 0.521107        | 0.806516 | 0.895342 |
| 4     | 0.371084      | 0.508541        | 0.824514 | 0.908062 |
| 5     | 0.317201      | 0.526333        | 0.827050 | 0.908501 |

_(Lưu ý: Đối với tác vụ Topic, Neuro-symbolic có chỉ số chênh lệch không đáng kể so với Baseline, hiệu suất cũng khá sát nhau)._

---

## 2. Tác vụ Actionability Classification (Phân loại Hành động)

### Mô hình Baseline (Đã giảm)

| Epoch | Training Loss | Validation Loss | Macro F1 | Micro F1 |
| ----- | ------------- | --------------- | -------- | -------- |
| 1     | 0.531650      | 0.527134        | 0.660718 | 0.782300 |
| 2     | 0.437036      | 0.509162        | 0.695363 | 0.812908 |
| 3     | 0.345409      | 0.550265        | 0.701787 | 0.813757 |
| 4     | 0.294554      | 0.616576        | 0.699019 | 0.809957 |
| 5     | 0.244115      | 0.687495        | 0.714741 | 0.827477 |

### Mô hình Neuro-Symbolic

| Epoch | Training Loss | Validation Loss | Macro F1 | Micro F1 |
| ----- | ------------- | --------------- | -------- | -------- |
| 1     | 0.635151      | 0.628616        | 0.709762 | 0.834160 |
| 2     | 0.478344      | 0.573343        | 0.761202 | 0.876317 |
| 3     | 0.398636      | 0.553570        | 0.768212 | 0.881897 |
| 4     | 0.348633      | 0.548058        | 0.775898 | 0.888097 |
| 5     | 0.283210      | 0.568695        | 0.778254 | 0.889027 |

_(Lưu ý: Có thể quan sát thấy mô hình Neuro-Symbolic ở tác vụ Actionability cho kết quả vượt trội và hội tụ ổn định hơn hơn hẳn so với Baseline)_.
