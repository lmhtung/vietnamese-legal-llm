# Các metric đánh giá Qwen3.5 Vietnamese Legal CPT

Tài liệu này mô tả toàn bộ metric được định nghĩa trong `metric.py`, cách các metric được sử dụng trong `finetune.py` và `evaluation.py`, công thức, ý nghĩa và hướng diễn giải kết quả.

## 1. Hai nhóm metric

Hệ thống hiện có hai nhóm metric riêng biệt.

### 1.1. Teacher-forced metrics

Các metric này được tính trực tiếp trên token ground truth:

- Token-weighted negative log-likelihood (NLL).
- Perplexity.
- Top-1 token accuracy.
- Top-5 token accuracy.
- Số token thực sự được chấm.

Chúng được dùng trong cả:

- Validation định kỳ khi chạy `finetune.py`.
- Đánh giá độc lập base model và finetuned model bằng `evaluation.py`.

Trong validation của `finetune.py`, checkpoint tốt nhất được chọn theo `eval_token_nll`, không phải `eval_loss`; xem [finetune.py](./finetune.py#L697).

### 1.2. Generation metrics

Các metric này yêu cầu model sinh continuation:

- ROUGE-1 F1.
- ROUGE-2 F1.
- Chunked ROUGE-L F1.
- BLEU-4.
- chrF++.
- Article-heading F1.
- Distinct-2.
- Repeated 4-gram ratio.
- EOS rate.
- Generated/reference length ratio.
- Thống kê độ dài và thời gian sinh.

Chúng chỉ được chạy trong `evaluation.py` khi bật `--generation`. Không chạy generation trên toàn bộ validation sau mỗi `eval_steps`, vì thao tác này tốn thời gian và VRAM đáng kể.

## 2. Causal language-model scoring

Với một document có chuỗi token:

\[
x_1,x_2,\ldots,x_T
\]

causal language model dùng các token đứng trước để dự đoán token tiếp theo:

\[
p_\theta(x_t\mid x_1,\ldots,x_{t-1})
\]

Vì vậy logits và labels phải được dịch một vị trí:

```python
shifted_logits = logits[..., :-1, :]
shifted_labels = labels[..., 1:]
```

Phép dịch được thực hiện tại [metric.py](./metric.py#L203). Token đầu tiên không được chấm vì không có token ngữ cảnh đứng trước trong document.

Các vị trí padding có label bằng `-100` bị loại khỏi toàn bộ phép tính tại [metric.py](./metric.py#L210). EOS thật ở cuối document vẫn được giữ lại và được chấm như một target token bình thường.

Trong `evaluation.py`, code bảo đảm document có đúng một EOS ở cuối tại [evaluation.py](./evaluation.py#L180).

## 3. Token-weighted NLL

### 3.1. Công thức

Negative log-likelihood trung bình trên token được định nghĩa là:

\[
\operatorname{NLL}
=
-\frac{1}{N}
\sum_{t\in V}
\log p_\theta(x_t\mid x_{<t})
\]

Trong đó:

- \(V\) là tập các vị trí label hợp lệ, không bao gồm padding `-100`.
- \(N=|V|\) là tổng số token được chấm.
- \(p_\theta(x_t\mid x_{<t})\) là xác suất model gán cho token ground truth.

Code cộng cross-entropy của từng batch với `reduction="sum"` tại [metric.py](./metric.py#L215).

Trên toàn bộ validation hoặc test set, NLL được tính theo:

\[
\operatorname{TokenNLL}
=
\frac{
\sum_d \operatorname{NLLSum}_d
}{
\sum_d N_d
}
\]

Phép cộng dồn được thực hiện tại [metric.py](./metric.py#L257) và phép chia cuối cùng tại [metric.py](./metric.py#L273).

### 3.2. Ý nghĩa

- NLL càng thấp càng tốt.
- NLL thấp nghĩa là model gán xác suất cao hơn cho token đúng.
- Vì đây là trung bình theo token, document dài đóng góp nhiều hơn document ngắn.
- Đây là metric chính để lựa chọn best checkpoint trong quá trình CPT.

### 3.3. Khác biệt giữa `eval_loss` và `eval_token_nll`

`eval_loss` do Hugging Face Trainer tự động trả về. Nó được tổng hợp từ loss của các evaluation batch và có thể chịu ảnh hưởng bởi cách document được ghép vào batch.

`eval_token_nll` do `metric.py` tính bằng cách cộng tổng NLL và tổng số token hợp lệ trên toàn bộ validation. Do các document pháp luật có độ dài rất khác nhau, `eval_token_nll` là đại lượng phù hợp hơn để so sánh checkpoint.

Trainer hiện sử dụng:

```python
metric_for_best_model="eval_token_nll"
greater_is_better=False
```

Xem [finetune.py](./finetune.py#L697).

## 4. Perplexity

### 4.1. Công thức

Perplexity được tính trực tiếp từ NLL:

\[
\operatorname{PPL}=\exp(\operatorname{NLL})
\]

Code nằm tại [metric.py](./metric.py#L276).

Trong code có:

```python
math.exp(min(token_nll, 80.0))
```

Giới hạn 80 chỉ để tránh overflow số học. Với kết quả hợp lệ thông thường, NLL nhỏ hơn rất nhiều so với 80.

### 4.2. Ý nghĩa

- Perplexity càng thấp càng tốt.
- Nếu hai model dùng cùng tokenizer và cùng tập dữ liệu, model có perplexity thấp hơn thường dự đoán văn bản domain đó tốt hơn.
- Không nên so sánh trực tiếp perplexity giữa hai model dùng tokenizer khác nhau.
- Perplexity không đánh giá trực tiếp tính đúng đắn pháp lý hay khả năng sinh cấu trúc hoàn chỉnh.

## 5. Top-1 token accuracy

### 5.1. Công thức

\[
\operatorname{Top1Acc}
=
\frac{1}{N}
\sum_{t\in V}
\mathbf{1}
\left[
\arg\max_v z_{t,v}=x_t
\right]
\]

Trong đó:

- \(z_{t,v}\) là logit của vocabulary token \(v\) tại vị trí \(t\).
- \(x_t\) là token ground truth.
- \(\mathbf{1}[\cdot]\) bằng 1 nếu điều kiện đúng, ngược lại bằng 0.

Code đếm dự đoán đúng tại [metric.py](./metric.py#L219), sau đó chuyển thành phần trăm tại [metric.py](./metric.py#L277).

### 5.2. Ý nghĩa

- Top-1 accuracy càng cao càng tốt.
- Đây là metric nghiêm ngặt: token ground truth phải là token có logit cao nhất.
- Accuracy có thể khó tăng mạnh vì nhiều vị trí văn bản có nhiều continuation hợp lý, trong khi dataset chỉ cung cấp một ground truth.

## 6. Top-5 token accuracy

### 6.1. Công thức

\[
\operatorname{Top5Acc}
=
\frac{1}{N}
\sum_{t\in V}
\mathbf{1}
\left[
x_t\in\operatorname{Top5}(z_t)
\right]
\]

Code lấy năm token có logit lớn nhất tại [metric.py](./metric.py#L220) và kiểm tra ground truth tại [metric.py](./metric.py#L223).

### 6.2. Ý nghĩa

- Top-5 accuracy càng cao càng tốt.
- Nó cho biết model có đặt token đúng trong nhóm các lựa chọn có xác suất cao hay không.
- Top-5 thường cao hơn top-1 và có thể phản ánh tiến bộ sớm hơn trong CPT.

## 7. Streaming validation metrics

Logits đầy đủ có kích thước xấp xỉ:

\[
\text{batch size}\times\text{sequence length}\times\text{vocabulary size}
\]

Với sequence dài tới 20.000 token, không nên giữ logits của toàn bộ validation trong RAM hoặc VRAM. `TokenMetricPreprocessor` xử lý logits theo các đoạn sequence nhỏ và nén mỗi batch thành bốn giá trị:

1. Tổng NLL.
2. Tổng số token hợp lệ.
3. Số token top-1 đúng.
4. Số token top-5 đúng.

Phần nén nằm tại [metric.py](./metric.py#L188). Phần cộng dồn streaming nằm tại [metric.py](./metric.py#L230).

Trainer bật:

```python
prediction_loss_only=False
batch_eval_metrics=True
```

Xem [finetune.py](./finetune.py#L699). Các callback được truyền vào `SFTTrainer` tại [finetune.py](./finetune.py#L719).

Tham số `--metric-logits-chunk-size` điều khiển số vị trí sequence được chuyển sang FP32 và chấm cùng lúc. Giảm giá trị này nếu validation bị thiếu VRAM.

## 8. Whole-document sliding-window scoring

Trong `evaluation.py`, teacher-forced metrics được tính trên toàn bộ document bằng các cửa sổ context có overlap. Mỗi target token chỉ được chấm đúng một lần.

Thuật toán nằm tại [metric.py](./metric.py#L295):

- Cửa sổ đầu tiên chấm các token từ vị trí thứ hai.
- Cửa sổ tiếp theo overlap với cửa sổ trước để cung cấp context.
- `scored_until` bảo đảm các token trong vùng overlap không bị cộng hai lần.
- `ppl_stride_tokens` quyết định số target token mới được chấm sau mỗi bước.
- `ppl_context_tokens` quyết định context tối đa model nhìn thấy khi dự đoán.

Các output tương ứng là:

- `full_document_nll`.
- `full_document_perplexity`.
- `full_document_top1_accuracy_percent`.
- `full_document_top5_accuracy_percent`.
- `full_document_scored_tokens`.

Việc tổng hợp toàn bộ document được thực hiện tại [metric.py](./metric.py#L482).

## 9. Cách tạo continuation để chấm generation

Code ưu tiên tách document ngay trước `Điều 2`:

- Prefix: phần mở đầu và `Điều 1`.
- Reference: nội dung bắt đầu từ `Điều 2`.

Nếu không tìm thấy `Điều 2`, code dùng một số token đầu làm prefix và phần sau làm reference. Logic nằm tại [metric.py](./metric.py#L361).

Base model và finetuned model luôn được chấm trên cùng danh sách document, cùng prefix và cùng reference. Danh sách được chọn một lần trong [evaluation.py](./evaluation.py#L189).

## 10. ROUGE-1 và ROUGE-2 F1

ROUGE-N so sánh n-gram giữa prediction \(P\) và reference \(R\).

### 10.1. Precision và recall

\[
P_n
=
\frac{
|\operatorname{Ngram}_n(P)\cap\operatorname{Ngram}_n(R)|
}{
|\operatorname{Ngram}_n(P)|
}
\]

\[
R_n
=
\frac{
|\operatorname{Ngram}_n(P)\cap\operatorname{Ngram}_n(R)|
}{
|\operatorname{Ngram}_n(R)|
}
\]

Phép giao sử dụng multiset, nghĩa là có xét số lần n-gram xuất hiện.

### 10.2. F1

\[
F1_n=\frac{2P_nR_n}{P_n+R_n}
\]

Hàm F1 dùng chung nằm tại [metric.py](./metric.py#L60). ROUGE-N được định nghĩa tại [metric.py](./metric.py#L77) và được gọi cho unigram/bigram tại [metric.py](./metric.py#L397).

### 10.3. Diễn giải

- ROUGE-1 đo độ trùng ở cấp từ.
- ROUGE-2 đo độ trùng của cặp từ liên tiếp và nhạy hơn với trật tự cục bộ.
- Điểm càng cao càng tốt.
- Một continuation đúng pháp lý nhưng diễn đạt khác reference vẫn có thể có ROUGE thấp.

## 11. Chunked ROUGE-L F1

ROUGE-L dựa trên longest common subsequence (LCS), tức dãy token chung dài nhất vẫn giữ nguyên thứ tự nhưng không bắt buộc liên tiếp.

Đặt:

\[
L=\operatorname{LCS}(P,R)
\]

Khi đó:

\[
P_{LCS}=\frac{L}{|P|}
\]

\[
R_{LCS}=\frac{L}{|R|}
\]

\[
F1_{LCS}
=
\frac{2P_{LCS}R_{LCS}}{P_{LCS}+R_{LCS}}
\]

Thuật toán LCS dùng hai hàng dynamic programming nằm tại [metric.py](./metric.py#L84).

Vì document pháp luật có thể rất dài, code chia prediction và reference thành các chunk 512 từ theo vị trí tương ứng. Kết quả được gộp bằng:

\[
\operatorname{ROUGE-L}_{chunked}
=
\frac{\sum_i w_iF1_i}{\sum_i w_i}
\]

với:

\[
w_i=\max(1,|P_i|,|R_i|)
\]

Logic chunking và weighted mean nằm tại [metric.py](./metric.py#L105).

## 12. BLEU-4

BLEU sử dụng modified n-gram precision từ bậc 1 đến 4.

Với mỗi bậc \(n\):

\[
p_n
=
\frac{
\sum_g\min(C_P(g),C_R(g))
}{
\sum_g C_P(g)
}
\]

Trong đó:

- \(C_P(g)\) là số lần n-gram \(g\) xuất hiện trong prediction.
- \(C_R(g)\) là số lần n-gram đó xuất hiện trong reference.
- Phép `min` ngăn model nhận điểm thêm do lặp một n-gram quá nhiều lần.

Đặt \(c\) là tổng số token prediction và \(r\) là tổng số token reference. Brevity penalty là:

\[
BP
=
\begin{cases}
1, & c>r\\
\exp(1-r/c), & c\le r
\end{cases}
\]

BLEU-4:

\[
\operatorname{BLEU}
=
100\times BP\times
\exp\left(
\frac{1}{4}\sum_{n=1}^{4}\log p_n
\right)
\]

Code sử dụng exponential smoothing khi một bậc n-gram không có match. Toàn bộ công thức nằm tại [metric.py](./metric.py#L414).

BLEU được tính ở cấp corpus, không lấy trung bình BLEU từng document. Điểm càng cao càng tốt.

## 13. chrF++

chrF++ đo overlap trên cả:

- Character n-gram bậc 1 đến 6.
- Word n-gram bậc 1 đến 2.

Với mỗi nhóm n-gram \(j\):

\[
P_j
=
\frac{\operatorname{matched\ ngrams}_j}
{\operatorname{predicted\ ngrams}_j}
\]

\[
R_j
=
\frac{\operatorname{matched\ ngrams}_j}
{\operatorname{reference\ ngrams}_j}
\]

Code lấy trung bình precision và recall của tám nhóm:

\[
\bar P=\frac{1}{8}\sum_{j=1}^{8}P_j
\]

\[
\bar R=\frac{1}{8}\sum_{j=1}^{8}R_j
\]

Sau đó dùng \(F_\beta\) với \(\beta=2\):

\[
F_\beta
=
100\times
\frac{(1+\beta^2)\bar P\bar R}
{\beta^2\bar P+\bar R}
\]

Code nằm tại [metric.py](./metric.py#L451).

chrF++ thường hữu ích với tiếng Việt vì nó ghi nhận mức độ tương đồng ở cấp ký tự, ngay cả khi tokenization theo từ không hoàn toàn trùng nhau. Điểm càng cao càng tốt.

## 14. Article-heading F1

Metric này trích các tiêu đề pháp luật dạng `Điều N`, bao gồm các dạng như `Điều 2`, `Điều 2a` hoặc số Điều có phân cấp.

Sau khi trích số Điều từ reference và prediction, code tạo multiset và tính:

\[
P_{article}
=
\frac{\text{số heading match}}
{\text{số heading trong prediction}}
\]

\[
R_{article}
=
\frac{\text{số heading match}}
{\text{số heading trong reference}}
\]

\[
F1_{article}
=
\frac{2P_{article}R_{article}}
{P_{article}+R_{article}}
\]

Regex và phép đếm nằm tại [metric.py](./metric.py#L157).

Metric này kiểm tra model có giữ được số và cấu trúc Điều hay không. Nó không đánh giá nội dung chi tiết bên trong từng Điều. Điểm càng cao càng tốt.

## 15. Distinct-2

Distinct-2 đo mức độ đa dạng của bigram trong output:

\[
\operatorname{Distinct2}
=
\frac{\text{số bigram khác nhau}}
{\text{tổng số bigram được sinh}}
\]

Code nằm tại [metric.py](./metric.py#L143).

Điểm cao thường thể hiện output ít lặp hơn. Tuy nhiên Distinct-2 quá cao không tự động có nghĩa là output tốt, vì văn bản ngẫu nhiên hoặc sai nội dung cũng có thể có nhiều bigram khác nhau.

## 16. Repeated 4-gram ratio

Với \(C(g)\) là số lần 4-gram \(g\) xuất hiện trong prediction:

\[
\operatorname{Repeat4}
=
\frac{
\sum_g\max(C(g)-1,0)
}{
\sum_g C(g)
}
\]

Code nằm tại [metric.py](./metric.py#L149).

- Metric càng thấp càng tốt.
- Giá trị cao cho thấy model đang lặp lại các cụm bốn từ.
- Metric này đặc biệt hữu ích để phát hiện degeneration trong long-form generation.

## 17. Length ratio

Với mỗi document:

\[
\operatorname{LengthRatio}
=
\frac{\text{generated tokens}}
{\text{reference tokens}}
\]

Code tại [evaluation.py](./evaluation.py#L374).

Diễn giải:

- Gần 1: prediction có độ dài tương đương reference.
- Nhỏ hơn 1 nhiều: model kết thúc sớm hoặc sinh thiếu nội dung.
- Lớn hơn 1 nhiều: model sinh dài quá mức, có thể lặp hoặc không kết thúc.

`average_length_ratio` là trung bình đều của ratio trên các document generation; xem [metric.py](./metric.py#L545).

## 18. EOS rate

Với \(D\) document được generation:

\[
\operatorname{EOSRate}
=
100\times
\frac{1}{D}
\sum_{d=1}^{D}
\mathbf{1}[\text{output}_d\text{ kết thúc bằng EOS}]
\]

Việc kiểm tra EOS của từng prediction nằm tại [evaluation.py](./evaluation.py#L370). Việc tổng hợp nằm tại [metric.py](./metric.py#L548).

EOS rate cao cho thấy model thường tự kết thúc. Tuy nhiên cần đọc cùng length ratio:

- EOS rate cao nhưng length ratio rất thấp có thể là kết thúc quá sớm.
- EOS rate thấp và length ratio cao có thể là model không biết dừng.

## 19. Các thống kê phụ

Các trường sau là thống kê phục vụ kiểm tra pipeline, không phải metric chất lượng độc lập:

- `num_documents`: số document đã được teacher-forced scoring.
- `num_generation_documents`: số document có generation hợp lệ.
- `full_document_scored_tokens`: tổng số target token được chấm.
- `average_prefix_tokens`: số token prefix trung bình.
- `average_reference_tokens`: số token reference trung bình.
- `average_generated_tokens`: số token output trung bình.
- `average_generation_seconds`: thời gian generation trung bình mỗi document.
- `total_generation_seconds`: tổng thời gian generation.

Các giá trị này được tổng hợp tại [metric.py](./metric.py#L482).

## 20. Cách aggregate metric

Teacher-forced metrics được gộp theo token:

\[
\frac{\sum_d\text{numerator}_d}
{\sum_d\text{token count}_d}
\]

Do đó document dài có trọng số lớn hơn.

Các metric generation per-document sau được lấy trung bình đều giữa các document rồi nhân 100:

- ROUGE-1 F1.
- ROUGE-2 F1.
- Chunked ROUGE-L F1.
- Article-heading F1.
- Distinct-2.
- Repeated 4-gram ratio.

Phần aggregate nằm tại [metric.py](./metric.py#L518).

BLEU và chrF++ được tính ở cấp toàn corpus trên toàn bộ prediction/reference, không phải trung bình từng document; xem [metric.py](./metric.py#L534).

## 21. So sánh base và finetuned model

`evaluation.py` chạy base model trước, giải phóng model khỏi GPU, rồi chạy finetuned adapter hoặc merged model trên đúng cùng tập document.

Với mỗi metric:

\[
\Delta
=
\operatorname{Metric}_{finetuned}
-
\operatorname{Metric}_{base}
\]

Code tính delta tại [evaluation.py](./evaluation.py#L433).

Đại lượng `improvement` được chuẩn hóa theo hướng tốt hơn:

- Với NLL, perplexity và repeated 4-gram ratio: `improvement = -delta`.
- Với các metric chất lượng còn lại: `improvement = delta`.

Quy tắc nằm tại [evaluation.py](./evaluation.py#L421).

## 22. Bảng hướng diễn giải

| Metric | Hướng tốt | Đánh giá chính |
|---|---:|---|
| Token NLL | Thấp hơn | Xác suất model gán cho token ground truth |
| Perplexity | Thấp hơn | Mức độ khó dự đoán corpus đối với model |
| Top-1 token accuracy | Cao hơn | Token đúng đứng đầu phân phối |
| Top-5 token accuracy | Cao hơn | Token đúng nằm trong năm lựa chọn đầu |
| ROUGE-1 F1 | Cao hơn | Overlap từ |
| ROUGE-2 F1 | Cao hơn | Overlap cặp từ và trật tự cục bộ |
| Chunked ROUGE-L F1 | Cao hơn | Dãy token chung giữ nguyên thứ tự |
| BLEU-4 | Cao hơn | Modified n-gram precision có phạt output ngắn |
| chrF++ | Cao hơn | Overlap character và word n-gram |
| Article-heading F1 | Cao hơn | Khả năng giữ cấu trúc và số Điều |
| Distinct-2 | Thường cao hơn | Độ đa dạng bigram; không nên đọc độc lập |
| Repeated 4-gram ratio | Thấp hơn | Mức lặp cụm từ trong output |
| EOS rate | Phụ thuộc | Khả năng tự kết thúc; phải đọc cùng length ratio |
| Length ratio | Thường gần 1 | Mức tương đồng về độ dài với reference |

## 23. Hạn chế cần lưu ý

1. NLL, perplexity và token accuracy không đo trực tiếp tính đúng đắn pháp lý.
2. ROUGE, BLEU và chrF++ chỉ so với một continuation reference; một output hợp pháp nhưng diễn đạt khác vẫn có thể bị chấm thấp.
3. Article-heading F1 chỉ nhìn số Điều, không kiểm tra nội dung Điều.
4. Distinct-2 cao không bảo đảm văn bản đúng hoặc mạch lạc.
5. Perplexity chỉ nên so sánh khi base và finetuned model sử dụng cùng tokenizer và cùng tập document.
6. `ppl_context_tokens` ảnh hưởng lượng context model nhìn thấy. Kết quả với context 8.192 và 20.000 token không hoàn toàn tương đương.
7. Sampling làm generation metric biến động. Mặc định `evaluation.py` dùng greedy decoding để việc so sánh có tính xác định.
8. Code hiện tại chưa tính BERTScore.

## 24. Metric được ghi ra khi finetune

Trong một scheduled validation, các trường quan trọng thường có dạng:

```text
eval_loss
eval_token_nll
eval_token_perplexity
eval_top1_token_accuracy_percent
eval_top5_token_accuracy_percent
eval_scored_tokens
```

Khi chạy baseline hoặc đánh giá best checkpoint, prefix lần lượt có thể là `baseline_*` và `best_*`.

Best checkpoint được chọn theo giá trị `eval_token_nll` thấp nhất.

## 25. Metric được ghi ra khi chạy `evaluation.py`

Kết quả teacher-forced:

```text
full_document_scored_tokens
full_document_nll
full_document_perplexity
full_document_top1_accuracy_percent
full_document_top5_accuracy_percent
```

Nếu bật generation, kết quả bổ sung gồm:

```text
rouge1_f1_percent
rouge2_f1_percent
rougeL_chunked_f1_percent
bleu
chrf_plus_plus
article_heading_f1_percent
distinct2_percent
repeated_4gram_ratio_percent
average_prefix_tokens
average_reference_tokens
average_generated_tokens
average_length_ratio
eos_rate_percent
average_generation_seconds
total_generation_seconds
```

Kết quả riêng của hai model nằm trong:

```text
<output-dir>/base/metrics.json
<output-dir>/finetuned/metrics.json
```

Bảng so sánh và delta nằm trong:

```text
<output-dir>/comparison.json
<output-dir>/report.md
```
