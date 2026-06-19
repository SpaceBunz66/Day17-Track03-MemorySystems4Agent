# Day 17 Memory Systems - Benchmark Analysis

## Kết quả benchmark

### Standard Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 2446 | 22492 | 0.018 | 0.021 | 0 | 0 |
| Advanced | 2721 | 33267 | 1.000 | 0.993 | 441 | 0 |

### Long-Context Stress Benchmark

| Agent | Agent tokens only | Prompt tokens processed | Cross-session recall | Response quality | Memory growth (bytes) | Compactions |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |
| Baseline | 570 | 28389 | 0.000 | 0.033 | 0 | 0 |
| Advanced | 710 | 10022 | 1.000 | 1.000 | 333 | 20 |

## Phân tích

Advanced recall tốt hơn Baseline vì các fact ổn định được ghi vào `User.md` theo `user_id`. Khi sang thread mới, Baseline chỉ có short-term memory của thread mới nên gần như không trả lời được các câu hỏi chéo phiên.

Ở benchmark chuẩn, Advanced xử lý nhiều prompt token hơn Baseline vì mỗi lượt phải kéo thêm profile memory. Đây là trade-off cố ý: hệ thống trả thêm chi phí ngữ cảnh để đổi lấy khả năng nhớ dài hạn và xử lý correction như đổi nơi ở hoặc nghề nghiệp.

Ở stress benchmark, compact memory tạo lợi thế rõ hơn. Baseline liên tục xử lý toàn bộ lịch sử dài nên prompt tokens tăng mạnh, còn Advanced nén phần cũ thành summary và chỉ giữ recent messages. Vì vậy Advanced giảm `Prompt tokens processed` từ 28389 xuống 10022 trong khi vẫn giữ recall chéo phiên.

File memory tăng trưởng là chi phí thật của persistent memory. Nếu không kiểm soát, `User.md` có thể phình to, chứa fact cũ hoặc fact sai. Bài làm hiện có bonus guardrail: chỉ ghi fact khi confidence đủ cao, bỏ qua câu hỏi/nhiễu rõ ràng, và upsert theo key để correction mới thay thế fact cũ.

## Bonus đã triển khai

- Confidence threshold heuristic: không lưu các giá trị nghi vấn như "đâu", "gì", "không".
- Structured entity extraction: facts được lưu theo key trong `User.md`.
- Conflict handling: `upsert_fact()` thay thế fact cũ khi có correction mới.
- Noise avoidance: bỏ qua các trường hợp câu đùa hoặc địa điểm không phải nơi ở hiện tại.
