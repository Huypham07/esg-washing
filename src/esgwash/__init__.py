"""esgwash — đo lường ESG-washing (talk-vs-walk) trong báo cáo ngân hàng VN.

Kiến trúc layered pipeline: data -> models -> grounding -> indices -> validation.
Mỗi bước là hàm thuần đọc/ghi artifact parquet, điều khiển bởi configs/*.yml.
Specs: superpowers/specs/00..05.
"""

__version__ = "0.1.0"
