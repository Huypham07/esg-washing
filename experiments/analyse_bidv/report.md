# Phân tích ESG-washing — BIDV (2023, 2024)

> Đơn vị = **chunk** (≤256 token). CTI = tỉ lệ cam kết "cheap talk". Specificity 3 mức: Mức 0 mơ hồ / Mức 1 cụ thể (hành động có tên) / Mức 2 định lượng. **CTI_loose** = chỉ Mức 0; **CTI_strict** = Mức 0+1 (chỉ định lượng mới tính thực chất). Sự thật nằm trong dải [loose, strict].

## 2023

- Chunk: **266** | ESG: **126** (47%) | commitment: **147** (55%) | commitment-ESG: **85**

### CTI tổng trên ESG (gộp trụ, đếm chunk duy nhất, n=85)
- Phân bố: Mức0=33 · Mức1=29 · Mức2=23
- **CTI_loose = 0.3882** · **CTI_strict = 0.7294**  → dải cheap-talk ESG

### CTI & selective disclosure theo trụ

| pillar | n_commit | lvl0 | lvl1 | lvl2 | cti_loose | cti_strict | n_esg_chunks | disclosure_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| env | 42 | 11 | 15 | 16 | 0.2619 | 0.619 | 47 | 0.2611 |
| soc | 41 | 14 | 12 | 15 | 0.3415 | 0.6341 | 48 | 0.2667 |
| gov | 51 | 25 | 18 | 8 | 0.4902 | 0.8431 | 85 | 0.4722 |


### gCTI-L1 (refine Mức 1 bằng grounding, θ_L1=0.4)
> L1 grounding chấm TỪNG hành động Mức 1 (có được corroborate trong báo cáo không) → thay vì đoán "tất cả Mức 1 thực chất" (loose) hay "tất cả cheap" (strict), gCTI-L1 là **điểm NẰM GIỮA dải [CTI_loose, CTI_strict]**. Band theo `l1_sim_floor`. (Mức 2 giữ substantive ở đây; full gCTI ground cả Mức 2 ở cti.parquet/compare_arms.) Proxy nhiễu — xem report P3-L1.

- **floor0.50** → gCTI-L1 (ESG) = **0.5765**  ·  env=0.4286 soc=0.4634 gov=0.7059
- **floor0.25** → gCTI-L1 (ESG) = **0.4941**  ·  env=0.3571 soc=0.439 gov=0.6471

### Ví dụ phân loại ĐÚNG (minh hoạ)

**Mức 2 — định lượng quy về BIDV**

| idx | lvl | E/S/G | text | ghi chú |
| --- | --- | --- | --- | --- |
| 4 | 2 | 1/0/0 | Dư nợ tín dụng đạt 1,75 triệu tỷ đồng, tăng trưởng Với vai trò là một định chế tài chính hàng đầu trong việc hỗ trợ phát triển kinh tế đất nước, BIDV kiên định theo đuổi mục tiêu trở thành "ngân hàng xanh" trong chiến lược kinh doanh giai đ | quant items: ['dư nợ tín dụng đạt 1,75 triệu tỷ đồng', 'chiếm khoảng 13% dư nợ tín dụng toàn nền kinh tế', 'cơ cấu tín dụng chuyển dịch theo hướng bền vững'] |
| 8 | 2 | 1/1/0 | hàng các sản phẩm tài chính bền vững theo chuẩn mực quốc tế; (iii) Tiên phong thu hút nguồn vốn phát triển bền vững, chuyển dịch kinh tế xanh, dẫn đầu thị trường về tài trợ tín dụng xanh; là ngân hàng đầu tiên phát hành thành công 2.500 tỷ  | quant items: ['phát hành thành công 2.500 tỷ đồng trái phiếu xanh theo tiêu chuẩn quốc tế'] |
| 24 | 2 | 1/0/0 | BIDV là ngân hàng thương mại cổ phần đầu tiên tại Việt Nam công bố "Khung Khoản vay bền vững" và hiện là ngân hàng dẫn đầu thị trường về tài trợ các dự án xanh với hàng ngàn khách hàng và dự án. Luôn nằm trong Top 5 ngân hàng có dư nợ tín d | quant items: ['dư nợ tín dụng xanh đạt 74.177 tỷ đồng', 'chiếm tỷ trọng 4,24% tổng dư nợ', 'chủ yếu tập trung vào lĩnh vực Năng lượng tái tạo, năng lượng sạch'] |
| 120 | 2 | 0/1/0 | Được hưởng các quyền lợi khác: mua cổ phần, cổ phiếu, tham gia các dự án kinh doanh bất động sản của BIDV. Được tổ chức Đảng cơ sở giúp đỡ, tạo điều kiện phấn đấu để được đứng trong hàng ngũ của Đảng (nếu có nguyện vọng). Với những chính sá | quant items: ['đào tạo nhân lực', 'đạt 111% kế hoạch số lớp', 'đạt 170% kế hoạch số học viên'] |
| 125 | 2 | 1/1/0 | Tiếp nối những hoạt động an sinh xã hội (ASXH) nhiều năm qua, công tác ASXH của BIDV năm 2023 tập trung vào các lĩnh vực chính theo định hướng của Chính phủ hướng tới phát triển bền vững bao gồm: giáo dục, y tế, xây nhà cho người nghèo, khắ | quant items: ['triển khai 158 chương trình an sinh xã hội'] |
| 126 | 2 | 0/1/0 | BIDV đã triển khai thực hiện tài trợ lĩnh vực giáo dục với tổng chí phí là gần 109,5 tỷ đồng, tài trợ xây dựng cơ sở vật chất, phòng học cho 07 cơ sở trường học, tặng hàng ngàn suất học bổng cho học sinh nghèo cả nước; Tài trợ các công trìn | quant items: ['tài trợ xây dựng cơ sở vật chất, phòng học cho 07 cơ sở trường học'] |


**Mức 1 — hành động/công cụ có tên**

| idx | lvl | E/S/G | text | ghi chú |
| --- | --- | --- | --- | --- |
| 9 | 1 | 0/0/0 | Đồng hành với hoạt động chuyển đổi số trong lĩnh vực tài chính ngân hàng, năm 2023 BIDV cũng đã tạo được những dấu ấn nổi bật khẳng định vị thế tiên phong trong hành trình chuyển đổi số: (i) Chính thức đi vào hoạt động hệ thống ngân hàng lõ | concrete items: ['chính thức đi vào hoạt động hệ thống ngân hàng lõi Core Banking Profile', 'tích hợp toàn diện hơn 100 ứng dụng vào hệ thống ngân hàng lõi', 'phát triển hệ thống Payment Hub'] |
| 17 | 1 | 1/0/0 | Xác định Chuyển đổi số và chuyển đổi xanh là xu hướng chuyển dịch tất yếu của thế giới và của Việt Nam, hướng tới phát triển bền vững, năm 2023 hoạt động ngân hàng Bán buôn của BIDV tiếp tục đi đầu trong việc triển khai các sản phẩm dịch vụ | concrete items: ['triển khai các sản phẩm dịch vụ ngân hàng số', 'triển khai sản phẩm tài chính Xanh', 'xây dựng hệ sinh thái ngân hàng số toàn diện'] |
| 19 | 1 | 0/0/0 | Việc phát triển hệ sinh thái số (digital ecosystem) cũng trở thành một trong những lĩnh vực được BIDV quan tâm. Ngày 29/11/2023, BIDV đã tiên phong ra mắt thị trường hệ thống BIDV Open API – hệ sinh thái mở giúp tích hợp các dịch vụ ngân hà | concrete items: ['ra mắt thị trường hệ thống BIDV Open API', 'tích hợp dịch vụ ngân hàng vào các ứng dụng, phần mềm, nền tảng số', 'phát triển sản phẩm theo hướng nâng cao trải nghiệm khách hàng'] |
| 20 | 1 | 0/0/0 | BIDV OPEN AI hệ sinh thái mở giúp tích hợp các dịch vụ ngân hàng vào các ứng dụng, phần mềm, nền tảng số của khách hàng, đối tác. Bên cạnh các ưu đãi về dịch vụ, BIDV luôn dành ưu tiên cấp tín dụng cho đối tượng doanh nghiệp xuất nhập khẩu  | concrete items: ['hệ sinh thái mở', 'tích hợp dịch vụ ngân hàng vào ứng dụng, phần mềm, nền tảng số', 'đầu tư vào sản phẩm tài trợ thương mại'] |
| 26 | 1 | 0/0/0 | ), đồng thời phát triển các tính năng mới đột phá như Smart Kids, sản phẩm đầu tư, bảo hiểm, chứng khoán đưa lên kênh SmartBanking. Đồng thời, liên tục hợp tác với các đối tác lớn, mở rộng và hoàn thiện hệ sinh thái tiện ích tài chính và ph | concrete items: ['phát triển các tính năng mới đột phá', 'liên tục hợp tác với các đối tác lớn', 'hoàn thiện hệ sinh thái tiện ích tài chính và phi tài chính'] |
| 27 | 1 | 1/0/0 | Hưởng ứng Chiến lược quốc gia về tăng trưởng xanh giai đoạn 2021 - 2030, tầm nhìn đến năm 2050, bên cạnh các gói vay ưu đãi như vay tiêu dùng, vay sản xuất kinh doanh thông thường, BIDV đã ban hành gói Tín dụng xanh dành cho khách hàng cá n | concrete items: ['ban hành gói Tín dụng xanh dành cho vay năng lượng sạch', 'tiến hành số hóa các quy trình, hồ sơ sản phẩm Tín dụng'] |


**Mức 0 — mơ hồ thật sự**

| idx | lvl | E/S/G | text | ghi chú |
| --- | --- | --- | --- | --- |
| 7 | 0 | 1/1/1 | Nhận thức được xu thế tất yếu của phát triển bền vững trong dài hạn, với vai trò là một định chế tài chính hàng đầu trong việc hỗ trợ phát triển kinh tế đất nước, BIDV đã kiên định theo đuổi mục tiêu trở thành "ngân hàng xanh" trong chiến l | đúng mơ hồ (không tên riêng/số) |
| 11 | 0 | 0/0/1 | Kết thúc năm 2023, BIDV cũng đạt được nhiều bước tiến trong quản trị điều hành, phát triển thể chế, tạo tiền đề vững chắc cho việc triển khai chiến lược kinh doanh giai đoạn 2021-2025 và các Chiến lược cấu phần, hướng tới phát triển bền vữn | đúng mơ hồ (không tên riêng/số) |
| 15 | 0 | 0/0/0 | Tập trung các biện pháp mở rộng quy mô hoạt động gắn với kiểm soát chặt chẽ chất lượng tín dụng, nâng cao chất lượng tài sản, cơ cấu lại nền khách hàng... Nâng cao hiệu quả hoạt động gắn với chuyển dịch cơ cấu thu nhập theo hướng gia tăng t | đúng mơ hồ (không tên riêng/số) |
| 29 | 0 | 0/0/0 | Thẻ tín dụng BIDV là một trong những dòng sản phẩm trọng tâm BIDV tập trung phát triển dựa trên các phân tích, nghiên cứu chuyên sâu về sở thích, thói quen chi tiêu, phong cách sống của khách hàng, gồm các dòng thẻ Cashback cho khách hàng t | đúng mơ hồ (không tên riêng/số) |
| 48 | 0 | 1/1/0 | Từng giữ các chức vụ: Phó Tổng Giám đốc Tổng Công ty Bảo hiểm BIDV; Giám đốc Ban Tài chính BIDV. Là định chế tài chính hàng đầu Việt Nam, BIDV luôn tiên phong đi đầu, thực thi có hiệu quả các chủ trương định hướng của Chính phủ, NHNN nhằm g | đúng mơ hồ (không tên riêng/số) |
| 49 | 0 | 1/1/1 | Tập trung nguồn lực xây dựng Chiến lược phát triển bền vững và thực hành ESG tổng thể tại BIDV nhằm đưa BIDV trở thành ngân hàng đứng đầu thị trường Việt Nam trong phát triển xanh, bền vững và thực hành ESG. Tích cực triển khai các hoạt độn | đúng mơ hồ (không tên riêng/số) |

### Ví dụ NGHI NHẦM (heuristic — cần kiểm tay)

**Nghi sai attribution (số của NHNN/quốc gia)**

| idx | lvl | E/S/G | text | ghi chú |
| --- | --- | --- | --- | --- |
| 2 | 2 | 0/0/0 | Trong năm, NHNN đã bốn lần giảm lãi suất điều hành với mức giảm 0,5-2% nhằm tháo gỡ khó khăn cho nền kinh tế, từ đó giúp giảm mặt bằng lãi suất huy động và cho vay; điều hành tỷ giá phù hợp; Hoàn thiện các quy định pháp lý trong hoạt động n | Mức 2 nhưng có NHNN/quốc gia -> số có thể KHÔNG của BIDV |
| 3 | 2 | 0/0/0 | Năm 2023 đối với BIDV có ý nghĩa quan trọng, là năm bản lề của Chiến lược phát triển kinh doanh giai đoạn 2021-2025, trong bối cảnh còn nhiều khó khăn, thách thức, với sự đoàn kết, đồng lòng, nỗ lực vượt khó của toàn hệ thống đã tạo nên một | Mức 2 nhưng có NHNN/quốc gia -> số có thể KHÔNG của BIDV |
| 4 | 2 | 1/0/0 | Dư nợ tín dụng đạt 1,75 triệu tỷ đồng, tăng trưởng Với vai trò là một định chế tài chính hàng đầu trong việc hỗ trợ phát triển kinh tế đất nước, BIDV kiên định theo đuổi mục tiêu trở thành "ngân hàng xanh" trong chiến lược kinh doanh giai đ | Mức 2 nhưng có NHNN/quốc gia -> số có thể KHÔNG của BIDV |
| 18 | 2 | 0/0/0 | Sản phẩm BIDV iBank và BIDV iConnect đã trở thành những trụ cột chính trong hệ sinh thái số của ngân hàng, góp phần quan trọng trong việc kiến tạo giá trị bền vững cho khách hàng. Đồng thời, nhằm đáp ứng mục tiêu chiến lược của Chính phủ tr | Mức 2 nhưng có NHNN/quốc gia -> số có thể KHÔNG của BIDV |
| 22 | 2 | 0/0/0 | BIDV cũng là ngân hàng đi đầu trong việc thực hiện các giải pháp tín dụng để góp phần thực hiện chủ trương, chỉ đạo của Đảng và Chính phủ trong việc hồi phục và phát triển nền kinh tế, qua việc triển khai Chương trình tín dụng 120.000 tỷ đồ | Mức 2 nhưng có NHNN/quốc gia -> số có thể KHÔNG của BIDV |
| 23 | 2 | 0/0/0 | Bằng khả năng cung cấp giải pháp "may đo" cho nhiều doanh nghiệp lớn, BIDV đã phối hợp xây dựng các chương trình tài trợ chuỗi cung ứng dành cho các tập đoàn hàng đầu trong các lĩnh vực kinh tế như bất động sản, hàng tiêu dùng nhanh (FMCG), | Mức 2 nhưng có NHNN/quốc gia -> số có thể KHÔNG của BIDV |


**Commitment bắt nhầm kết quả tài chính**

| idx | lvl | E/S/G | text | ghi chú |
| --- | --- | --- | --- | --- |
| 3 | 2 | 0/0/0 | Năm 2023 đối với BIDV có ý nghĩa quan trọng, là năm bản lề của Chiến lược phát triển kinh doanh giai đoạn 2021-2025, trong bối cảnh còn nhiều khó khăn, thách thức, với sự đoàn kết, đồng lòng, nỗ lực vượt khó của toàn hệ thống đã tạo nên một | commitment nhưng là KẾT QUẢ tài chính (không trụ ESG) |
| 5 | 2 | 0/0/0 | Hiệu quả kinh doanh tăng trưởng tích cực so với cùng kỳ năm trước, đạt và vượt kế hoạch năm 2023 đã đề ra: (i) Chênh lệch thu chi đạt 47.932 tỷ đồng. Lợi nhuận trước thuế hợp nhất đạt 27.589 tỷ đồng, tăng trưởng 20,4%, vượt kế hoạch ĐHĐCĐ g | commitment nhưng là KẾT QUẢ tài chính (không trụ ESG) |
| 31 | 1 | 0/0/0 | Bên cạnh gia tăng quy mô tiền gửi có kỳ hạn, BIDV cũng triển khai mạnh mẽ các giải pháp thu hút khách hàng sử dụng tài khoản BIDV làm tài khoản chính, từ đó gia tăng nguồn tiền gửi không kỳ hạn (CASA) của khách hàng như dịch vụ Tài khoản Nh | commitment nhưng là KẾT QUẢ tài chính (không trụ ESG) |
| 102 | 0 | 0/0/0 | BIC thực hiện IPO thành công, chính thức chuyển sang mô hình công ty cổ phần từ ngày 01/10/2010, niêm yết cổ phiếu trên Sàn Giao dịch chứng khoán TP. Hồ Chí Minh năm 2011 và bán chiến lược cho Fairfax Asia Limited - Công ty con của Fairfax  | commitment nhưng là KẾT QUẢ tài chính (không trụ ESG) |
| 110 | 2 | 0/0/0 | Năm 2009, BAMC hoàn tất quá trình cơ cấu lại hoạt động theo hướng duy trì pháp nhân, thu gọn tối đa hoạt động kinh doanh và nhân sự. Được sự phê duyệt của NHNN tại Công văn số 40/NHNN-TTGSNH ngày 03/01/2018 về việc tái cơ cấu BAMC, HĐQT BID | commitment nhưng là KẾT QUẢ tài chính (không trụ ESG) |
| 112 | 2 | 0/0/0 | Tuy nhiên, VRB chủ động triển khai các giải pháp xử lý các khó khăn hiệu quả, thay đổi định hướng hoạt động và tìm kiếm cơ hội kinh doanh mới để gia tăng thu nhập nên đạt được kết quả kinh doanh khá ấn tượng, cụ thể: Huy động vốn từ TCKT, d | commitment nhưng là KẾT QUẢ tài chính (không trụ ESG) |


**Mức 0 bỏ sót tên riêng (đáng lẽ Mức 1)**

| idx | lvl | E/S/G | text | ghi chú |
| --- | --- | --- | --- | --- |
| 12 | 0 | 0/0/0 | ấn với việc xác lập hợp tác chiến lược với Edmond de Rothschild - định chế tài chính hàng đầu thế giới về đầu tư và quản lý tài sản, đem tới cho khách hàng cá nhân cao cấp cơ hội sử dụng dịch vụ tài chính đẳng cấp toàn cầu; (iv) Tiếp tục đổ | Mức 0 nhưng CÓ tên riêng -> đáng lẽ Mức 1 |
| 14 | 0 | 1/1/1 | hiệ̣u mạnh Việt Nam 2023 (Vietnam Economic Times). Năm 2024 là năm tăng tốc, bứt phá, có ý nghĩa đặc biệt quan trọng trong việc thực hiện thắng lợi mục tiêu Chiến lược kinh doanh 5 năm. BIDV xác định phương châm hành động của năm là "Tinh g | Mức 0 nhưng CÓ tên riêng -> đáng lẽ Mức 1 |
| 21 | 0 | 0/0/0 | Danh mục sản phẩm tài trợ thương mại (TTTM) của BIDV đầy đủ, đa dạng, cơ chế sản phẩm tiêu chuẩn theo thông lệ quốc tế như: Thư tín dụng (L/C), nhờ thu, chiết khấu hối phiếu đòi nợ, Forfaiting. Mạng lưới ngân hàng đại lý/thị trường thanh to | Mức 0 nhưng CÓ tên riêng -> đáng lẽ Mức 1 |
| 72 | 0 | 0/0/1 | Tính toán, giám sát và kiểm tra sức chịu đựng về vốn yêu cầu cho RRHĐ; nghiên cứu triển khai, thí điểm tính vốn yêu cầu (VYC) cho RRHĐ theo Basel III. Tăng cường thực hiện các báo cáo chuyên đề cảnh báo rủi ro, thực hiện các báo cáo định kỳ | Mức 0 nhưng CÓ tên riêng -> đáng lẽ Mức 1 |
| 75 | 0 | 0/0/1 | Trong năm 2024, BIDV sẽ tiếp tục triển khai đầy đủ các công việc QLRRTT bảo đảm tuân thủ quy định của NHNN, quy định nội bộ; đồng thời đẩy mạnh nghiên cứu thông lệ, bám sát lộ trình triển khai Basel III của NHNN, tập trung chuyển đổi số, nâ | Mức 0 nhưng CÓ tên riêng -> đáng lẽ Mức 1 |
| 185 | 0 | 0/0/1 | Nâng cao năng lực quản trị điều hành, phát huy vai trò, trách nhiệm các cấp, đặc biệt là tinh thần gương mẫu, nêu gương của người đứng đầu đơn vị; duy trì kỷ cương, kỷ luật hành chính, cụ thể hóa trách nhiệm của từng cá nhân, bộ phận trong  | Mức 0 nhưng CÓ tên riêng -> đáng lẽ Mức 1 |


## 2024

- Chunk: **301** | ESG: **139** (46%) | commitment: **164** (55%) | commitment-ESG: **84**

### CTI tổng trên ESG (gộp trụ, đếm chunk duy nhất, n=84)
- Phân bố: Mức0=33 · Mức1=22 · Mức2=29
- **CTI_loose = 0.3929** · **CTI_strict = 0.6548**  → dải cheap-talk ESG

### CTI & selective disclosure theo trụ

| pillar | n_commit | lvl0 | lvl1 | lvl2 | cti_loose | cti_strict | n_esg_chunks | disclosure_share |
| --- | --- | --- | --- | --- | --- | --- | --- | --- |
| env | 39 | 12 | 8 | 19 | 0.3077 | 0.5128 | 48 | 0.25 |
| soc | 42 | 15 | 10 | 17 | 0.3571 | 0.5952 | 49 | 0.2552 |
| gov | 50 | 25 | 15 | 10 | 0.5 | 0.8 | 95 | 0.4948 |


### gCTI-L1 (refine Mức 1 bằng grounding, θ_L1=0.4)
> L1 grounding chấm TỪNG hành động Mức 1 (có được corroborate trong báo cáo không) → thay vì đoán "tất cả Mức 1 thực chất" (loose) hay "tất cả cheap" (strict), gCTI-L1 là **điểm NẰM GIỮA dải [CTI_loose, CTI_strict]**. Band theo `l1_sim_floor`. (Mức 2 giữ substantive ở đây; full gCTI ground cả Mức 2 ở cti.parquet/compare_arms.) Proxy nhiễu — xem report P3-L1.

- **floor0.50** → gCTI-L1 (ESG) = **0.5833**  ·  env=0.4359 soc=0.5476 gov=0.74
- **floor0.25** → gCTI-L1 (ESG) = **0.5238**  ·  env=0.3846 soc=0.4524 gov=0.68

### Ví dụ phân loại ĐÚNG (minh hoạ)

**Mức 2 — định lượng quy về BIDV**

| idx | lvl | E/S/G | text | ghi chú |
| --- | --- | --- | --- | --- |
| 11 | 2 | 1/1/1 | Phối hợp ADB đánh giá kết quả dự án ngân hàng xanh, ban hành Khung trái phiếu bền vững và ký kết Thỏa ước hạn mức tín dụng khí hậu với AFD; (ii) Dẫn đầu thị trường với hệ sinh thái sản phẩm tài chính xanh, tạo động lực thúc đẩy nền kinh tế  | quant items: ['5.000 tỷ đồng tiền gửi xanh', '3.000 tỷ đồng trái phiếu bền vững', '10.000 tỷ đồng đã được phân bổ cho các công trình xanh'] |
| 25 | 2 | 1/0/0 | Tiếp nối thành công của Trái phiếu xanh, năm 2024 ghi dấu mốc quan trọng khi BIDV lần đầu tiên triển khai sản phẩm Tiền gửi xanh. Sản phẩm đã nhận được sự hưởng ứng tích cực từ cộng đồng doanh nghiệp khi chỉ trong hơn 2 tháng, BIDV đã huy đ | quant items: ['triển khai sản phẩm Tiền gửi xanh', 'dư nợ tín dụng xanh'] |
| 26 | 2 | 1/0/0 | Năm 2024, BIDV tiếp tục triển khai thành công nhiều gói tín dụng xanh như: gói tín dụng Dệt may xanh (quy mô 4.200 tỷ đồng); gói tài trợ dự án Công trình xanh (quy mô 10.000 tỷ đồng); chương trình tín dụng xanh tài trợ Dự án sản xuất và cun | quant items: ['gói tín dụng Dệt may xanh', 'gói tài trợ dự án Công trình xanh', 'chương trình tín dụng xanh tài trợ Dự án sản xuất và cung cấp nước sạch'] |
| 27 | 2 | 1/0/0 | Năm 2024, bên cạnh các Nhà tài trợ AFD, ADB, EDCF, EIB, BIDV tiếp tục đẩy mạnh quan hệ với các Nhà tài trợ mới KOICA, DFAT, SCI, CTFK, IDE, huy động thành công 17 nguồn vốn mới với tổng giá trị 265 triệu USD (tương đương 6.500 tỷ đồng). Tro | quant items: ['huy động thành công 17 nguồn vốn mới với tổng giá trị 265 triệu USD', 'nguồn vốn ủy thác nước ngoài tài trợ lĩnh vực xanh, phát triển bền vững đạt 183 triệu USD'] |
| 32 | 2 | 1/0/0 | Đồng thời, ngân hàng liên tục hợp tác với các đối tác lớn để mở rộng và hoàn thiện hệ sinh thái tiện ích tài chính và phi tài chính hấp dẫn, đa dạng cho khách hàng như Data 4G, Voucher Dealtoday, Mua sắm hoàn tiền, Vietlott SMS, Dịch vụ Gol | quant items: ['trồng 1 triệu cây xanh thông qua trò chơi Green Mission'] |
| 33 | 2 | 1/1/0 | Hưởng ứng Chiến lược quốc gia về tăng trưởng xanh giai đoạn 2021 - 2030, tầm nhìn đến năm 2050, tín dụng xanh, bền vững được BIDV xác định là một trong các mục tiêu ưu tiên hàng đầu trong chiến lược kinh doanh dài hạn. Song song với đó là h | quant items: ['triển khai Gói tín dụng 10.000 tỷ đồng'] |


**Mức 1 — hành động/công cụ có tên**

| idx | lvl | E/S/G | text | ghi chú |
| --- | --- | --- | --- | --- |
| 9 | 1 | 0/0/1 | BIDV tiếp tục triển khai các công cụ quản lý rủi ro hoạt động và thị trường; chủ động triển khai lộ trình áp dụng Basel III, phương pháp xếp hạng nội bộ (FIRB), và áp dụng Chuẩn mực Báo cáo tài chính quốc tế số 9 (IFRS9) nhằm hiện đại hóa h | concrete items: ['triển khai các công cụ quản lý rủi ro hoạt động và thị trường', 'chủ động triển khai lộ trình áp dụng Basel III', 'áp dụng Chuẩn mực Báo cáo tài chính quốc tế số 9 (IFRS9)'] |
| 10 | 1 | 1/1/1 | Ngân hàng triển khai các nền tảng số tiên tiến, tích hợp dữ liệu và phân tích AI trong hoạt động kinh doanh, đồng thời nâng cấp hệ thống bảo mật và hạ tầng CNTT đáp ứng tiêu chuẩn quốc tế: Hệ thống ngân hàng lõi Corebanking được tối ưu hóa  | concrete items: ['triển khai các nền tảng số tiên tiến, tích hợp dữ liệu và phân tích AI', 'nâng cấp hệ thống bảo mật và hạ tầng CNTT', 'tối ưu hóa hệ thống ngân hàng lõi Corebanking với 165 tính năng mới tự phát triển'] |
| 22 | 1 | 0/0/0 | BIDV iBank – hệ thống ngân hàng điện tử cung cấp toàn diện các dịch vụ ngân hàng trên cả hai kênh app/web với cơ chế xử lý tự động, tức thời, 24/7. BIDV iConnect - giải pháp tích hợp liền mạch dịch vụ ngân hàng với hệ thống ERP của khách hà | concrete items: ['cung cấp toàn diện các dịch vụ ngân hàng trên cả hai kênh app/web', 'xử lý tự động, tức thời, 24/7', 'giải pháp tích hợp liền mạch dịch vụ ngân hàng với hệ thống ERP của khách hàng'] |
| 31 | 1 | 0/0/0 | BIDV không ngừng đổi mới, sáng tạo và phát triển các sản phẩm, dịch vụ tài chính đột phá, đồng thời nâng cao trải nghiệm khách hàng qua những giải pháp số hóa tiên tiến. Với chiến lược vững chắc, BIDV không chỉ duy trì sự tin tưởng của khác | concrete items: ['phát triển các sản phẩm, dịch vụ tài chính đột phá', 'nâng cao trải nghiệm khách hàng qua giải pháp số hóa tiên tiến'] |
| 39 | 1 | 0/0/0 | BIDV cũng là ngân hàng đầu tiên tổ chức chuỗi sự kiện chuyên biệt về đầu tư kết hợp với các đối tác trong và ngoài nước như Global Insight, Elevation Talks, Investor Days cung cấp những phân tích chuyên sâu về kinh tế vĩ mô cho khách hàng c | concrete items: ['tổ chức chuỗi sự kiện chuyên biệt về đầu tư', 'phối hợp tổ chức sự kiện đầu tư kết hợp thưởng rượu vang', 'khẳng định năng lực cung cấp dịch vụ Private Banking'] |
| 41 | 1 | 0/0/1 | Dự án chuyển đổi số quản trị nội bộ toàn hàng (B.One) do 100% nguồn lực nội bộ của BIDV tự xây dựng, go-live từ ngày 01/07/2024. B.One đã thay đổi căn bản cách thức vận hành hoạt động quản trị nội bộ của toàn hệ thống BIDV. Với quy mô lớn v | concrete items: ['ban hành và triển khai B.One', 'thay đổi cách thức vận hành quản trị nội bộ'] |


**Mức 0 — mơ hồ thật sự**

| idx | lvl | E/S/G | text | ghi chú |
| --- | --- | --- | --- | --- |
| 1 | 0 | 0/0/0 | Các chi nhánh và các đơn vị trực thuộc Công ty con, công ty liên doanh, liên kết Năm 2024 là năm có ý nghĩa đặc biệt quan trọng đối với BIDV, là năm toàn hệ thống "tăng tốc, bứt phá" nhằm thực hiện thắng lợi mục tiêu Chiến lược kinh doanh 5 | đúng mơ hồ (không tên riêng/số) |
| 8 | 0 | 0/0/1 | BIDV tiếp tục hoàn thiện thể chế, cập nhật kịp thời các cơ chế, chính sách theo quy định pháp luật mới nhằm nâng cao năng lực phản ứng chính sách, tháo gỡ điểm nghẽn và tối ưu hóa nguồn lực thúc đẩy kinh doanh: (i) Không ngừng nâng cao năng | đúng mơ hồ (không tên riêng/số) |
| 12 | 0 | 1/1/0 | trách nhiệm xã hội ý nghĩa như: giải chạy "BIDVRUN - Cho cuộc sống Xanh", "Tết ấm cho người nghèo", chương trình "Nước ngọt cho cuộc sống xanh", xây dựng Nhà văn hóa cộng đồng tránh lũ, hỗ trợ khắc phục thiên tai. Đặc biệt, BIDV đã tổ chức  | đúng mơ hồ (không tên riêng/số) |
| 13 | 0 | 0/0/0 | Theo đó, năm 2024, BIDV tiếp tục được các tổ chức, cộng đồng đánh giá cao với những giải thưởng uy tín: "Thương hiệu quốc gia" lần thứ 8; "Top 1000 doanh nghiệp niêm yết lớn nhất thế giới"; "Top 10 ngân hàng lớn nhất Đông Nam Á"; "Top 50 Cô | đúng mơ hồ (không tên riêng/số) |
| 14 | 0 | 0/0/0 | Năm 2025 là năm có ý nghĩa đặc biệt quan trọng đối với đất nước, đánh dấu thời điểm hoàn thành các mục tiêu, chỉ tiêu kế hoạch phát triển kinh tế - xã hội giai đoạn 2021-2025, đồng thời mở ra một kỷ nguyên mới - kỷ nguyên vươn mình của dân  | đúng mơ hồ (không tên riêng/số) |
| 15 | 0 | 0/0/1 | Theo đó, toàn hệ thống sẽ tập trung vào các trọng tâm sau: (i) Tăng cường trách nhiệm cá nhân và tập thể trong thực hiện chức trách, nhiệm vụ được giao, đảm bảo hiệu quả triển khai đồng bộ trong toàn hệ thống; (ii) Tổ chức tinh gọn từ bộ má | đúng mơ hồ (không tên riêng/số) |

### Ví dụ NGHI NHẦM (heuristic — cần kiểm tay)

**Nghi sai attribution (số của NHNN/quốc gia)**

| idx | lvl | E/S/G | text | ghi chú |
| --- | --- | --- | --- | --- |
| 23 | 2 | 0/0/0 | Đặc biệt, BIDV Open API đã ghi dấu ấn nổi bật khi thu hút gần 260.000 lượt gọi API thử nghiệm, gần 1.000 tài khoản trải nghiệm, 165 đối tác đăng ký tích hợp, 55 đối tác ký hợp đồng hợp tác, hơn 3.000 khách hàng doanh nghiệp sử dụng dịch vụ  | Mức 2 nhưng có NHNN/quốc gia -> số có thể KHÔNG của BIDV |
| 30 | 2 | 0/0/0 | BIDV luôn đi đầu trong thực hiện chủ trương, đường lối của Đảng và Nhà nước nhằm phục hồi và phát triển nền kinh tế bằng các giải pháp tín dụng như: Chương trình tín dụng nhằm tháo gỡ khó khăn, thúc đẩy sản xuất, xuất khẩu đối với doanh ngh | Mức 2 nhưng có NHNN/quốc gia -> số có thể KHÔNG của BIDV |
| 33 | 2 | 1/1/0 | Hưởng ứng Chiến lược quốc gia về tăng trưởng xanh giai đoạn 2021 - 2030, tầm nhìn đến năm 2050, tín dụng xanh, bền vững được BIDV xác định là một trong các mục tiêu ưu tiên hàng đầu trong chiến lược kinh doanh dài hạn. Song song với đó là h | Mức 2 nhưng có NHNN/quốc gia -> số có thể KHÔNG của BIDV |
| 118 | 2 | 0/0/0 | Được sự phê duyệt của NHNN tại Công văn số 40/NHNNTTGSNH ngày 03/01/2018 về việc tái cơ cấu BAMC, HĐQT BIDV đã có Quyết định số 189/NQ-BIDV ngày 12/04/2018 về việc tăng vốn điều lệ cho BAMC lên 100 tỷ đồng, nhằm hỗ trợ Công ty bước đầu triể | Mức 2 nhưng có NHNN/quốc gia -> số có thể KHÔNG của BIDV |
| 124 | 2 | 0/0/1 | VRB luôn đảm bảo tuân thủ các tỷ lệ an toàn theo quy định, công tác quản trị rủi ro tiếp tục được kiện toàn phù hợp với chuẩn mực Basel II và thông lệ. Nhằm góp phần phát triển ngành hàng không quốc gia, trên cơ sở phê duyệt của Thủ tướng C | Mức 2 nhưng có NHNN/quốc gia -> số có thể KHÔNG của BIDV |
| 125 | 2 | 0/0/0 | Năm 2024, VALC tiếp tục hoàn thành kế hoạch được giao với tổng doanh thu đạt 71,3 triệu USD, bằng 109% kế hoạch; lợi nhuận trước thuế đạt 18,8 triệu USD, bằng 125% kế hoạch năm. Trong năm 2024, hoạt động góp vốn, mua cổ phần tại BIDV tập tr | Mức 2 nhưng có NHNN/quốc gia -> số có thể KHÔNG của BIDV |


**Commitment bắt nhầm kết quả tài chính**

| idx | lvl | E/S/G | text | ghi chú |
| --- | --- | --- | --- | --- |
| 3 | 2 | 0/0/0 | Hiệu quả kinh doanh tăng trưởng tích cực so với cùng kỳ năm trước, đạt và vượt kế hoạch năm 2024 đã đề ra: (i) Chênh lệch thu chi hợp nhất đạt 53.094 tỷ đồng. Lợi nhuận trước thuế hợp nhất đạt 31.985 tỷ đồng, tăng trưởng 15,9%, vượt kế hoạc | commitment nhưng là KẾT QUẢ tài chính (không trụ ESG) |
| 4 | 2 | 0/0/0 | BIDV cũng luôn là một trong những ngân hàng đóng có góp tích cực vào ngân sách nhà nước với mức đóng góp trong năm đạt gần 9.300 tỷ đồng. Năm 2024, với nỗ lực quyết tâm cao độ, BIDV đã triển khai quyết liệt các phương án tăng vốn điều lệ, v | commitment nhưng là KẾT QUẢ tài chính (không trụ ESG) |
| 20 | 2 | 0/0/0 | Sáp nhập thành công Ngân hàng TMCP Phát triển Nhà đồng bằng Sông Cửu Long (MHB) vào hệ thống BIDV Ký kết thỏa thuận hợp tác chiến lược và công bố KEB Hana Bank (Hàn Quốc) là cổ đông chiến lược nước ngoài sở hữu 15% vốn điều lệ của BIDV. Hoà | commitment nhưng là KẾT QUẢ tài chính (không trụ ESG) |
| 38 | 0 | 0/0/0 | Nổi bật là BIDV Women &amp; Wealth, P-Fund và CD Flex, giúp tối ưu dòng tiền, nâng cao hiệu quả đầu tư và đảm bảo lợi nhuận bền vững: BIDV Women &amp; Wealth – giải pháp tài chính toàn diện dành cho nữ doanh nhân và khách hàng nữ thành đạt, | commitment nhưng là KẾT QUẢ tài chính (không trụ ESG) |
| 48 | 0 | 0/0/0 | Hội thi khuyến khích các cán bộ BIDV sáng tạo, xây dựng các ý tưởng mới về sản phẩm số, giải pháp công nghệ chưa được triển khai tại BIDV, có tiềm năng lớn ở thị trường Việt Nam; đồng thời phát huy tối đa khả năng phối hợp của đội ngũ, năng | commitment nhưng là KẾT QUẢ tài chính (không trụ ESG) |
| 117 | 0 | 0/0/0 | BAMC được thành lập năm 2001, hoạt động chính tập trung vào việc nhận và xử lý các khoản nợ của BIDV phát sinh trước thời điểm 31/12/2000. Chỉ sau 7 năm hoạt động, BAMC cơ bản hoàn thành công tác xử lý nợ xấu theo Quyết định số 149/2001/ QĐ | commitment nhưng là KẾT QUẢ tài chính (không trụ ESG) |


**Mức 0 bỏ sót tên riêng (đáng lẽ Mức 1)**

| idx | lvl | E/S/G | text | ghi chú |
| --- | --- | --- | --- | --- |
| 38 | 0 | 0/0/0 | Nổi bật là BIDV Women &amp; Wealth, P-Fund và CD Flex, giúp tối ưu dòng tiền, nâng cao hiệu quả đầu tư và đảm bảo lợi nhuận bền vững: BIDV Women &amp; Wealth – giải pháp tài chính toàn diện dành cho nữ doanh nhân và khách hàng nữ thành đạt, | Mức 0 nhưng CÓ tên riêng -> đáng lẽ Mức 1 |
| 70 | 0 | 0/0/1 | Tăng cường thực hiện các báo cáo chuyên đề cảnh báo rủi ro, thực hiện các báo cáo định kỳ, đột xuất về công tác QLRRHĐ. Tăng cường số hóa, xây dựng chương trình QLRRHĐ. Đào tạo, truyền thông về công tác QLRRHĐ trong toàn hệ thống. Trong năm | Mức 0 nhưng CÓ tên riêng -> đáng lẽ Mức 1 |
| 73 | 0 | 0/0/1 | Trong năm 2025, BIDV sẽ tiếp tục triển khai đầy đủ các công việc QLRRTT bảo đảm tuân thủ quy định của NHNN, quy định nội bộ; đồng thời bám sát lộ trình triển khai Basel III của NHNN, tập trung chuyển đổi số, nâng cấp, cập nhật hệ thống phần | Mức 0 nhưng CÓ tên riêng -> đáng lẽ Mức 1 |
| 168 | 0 | 0/0/1 | Tập trung nâng cao năng lực quản lý rủi ro; hoàn thành và đưa vào triển khai chương trình cảnh báo sớm rủi ro tín dụng, góp phần kịp thời phát hiện các rủi ro tín dụng có thể xảy ra, giảm thiểu tổn thất tín dụng; tiếp tục nghiên cứu cải tiế | Mức 0 nhưng CÓ tên riêng -> đáng lẽ Mức 1 |


## Hình

- `cti_band.png` — CTI_loose vs CTI_strict theo trụ
- `spec_level.png` — phân bố 3 mức specificity theo trụ
- `selective_disclosure.png` — số chunk ESG mỗi trụ (né chủ đề khó?)
