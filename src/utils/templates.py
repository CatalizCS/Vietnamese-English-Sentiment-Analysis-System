import random

class CommentTemplates:
    def __init__(self):
        self.vi_templates = {
            "product_review": {
                "positive": [
                    "Sản phẩm {quality} quá, {reason}",
                    "Mình rất {emotion} với {aspect}",
                    "Đóng gói {packaging}, giao hàng {delivery}",
                    "{aspect} tốt hơn mong đợi, {details}",
                    "Giá tiền {price_opinion}, {value_desc}",
                ],
                "negative": [
                    "{aspect} không được tốt, {issue}",
                    "Hơi thất vọng về {aspect}, {reason}",
                    "Chất lượng {quality_issue}, {details}",
                    "Giao hàng {delivery_issue}, {problem}",
                    "Không đáng giá tiền, {reason}",
                ],
                "neutral": [
                    "Sản phẩm tạm được, {details}",
                    "Cũng được, nhưng {suggestion}",
                    "Không có gì đặc biệt, {reason}",
                    "Dùng được, {opinion}",
                    "Tạm ổn, {details}",
                ],
            },
            "food_review": {
                "positive": [
                    "Món này {taste} thiệt luôn, {details}",
                    "Quán {service} chu đáo, {atmosphere}",
                    "Đồ ăn {quality} xuất sắc, {reason}",
                    "Giá cả {price_opinion}, {value}",
                    "Không gian {atmosphere}, {recommendation}",
                ],
                "negative": [
                    "Đồ ăn {taste_issue}, {reason}",
                    "Phục vụ {service_issue}, {problem}",
                    "Giá hơi {price_complaint}, {details}",
                    "Vệ sinh {cleanliness_issue}, {problem}",
                    "Thất vọng về {aspect}, {reason}",
                    "Đồ ăn {taste_issue} vl, {problem}",
                    "Quán này {slang_negative} thật, {reason}",
                    "Giá thì cắt cổ mà đồ ăn {taste_issue} vcl",
                    "Phục vụ như c*t, {service_issue}",
                    "Vệ sinh {cleanliness_issue} vl, éo bao giờ quay lại"
                ],
                "neutral": [
                    "Đồ ăn {taste} bình thường, {details}",
                    "Giá tương đối {price_opinion}, {value}",
                    "Phục vụ {service} tạm được, {details}",
                    "Không gian {atmosphere}, nhưng {suggestion}",
                    "Chất lượng trung bình, {opinion}"
                ],
            },
            "movie_review": {
                "positive": [
                    "Phim hay quá, {reason}",
                    "Diễn viên {acting}, kịch bản {script}",
                    "Cốt truyện {plot_opinion}, {details}",
                    "Xem mà {emotion}, {reason}",
                    "Đáng xem nha mọi người, {recommendation}",
                ],
                "negative": [
                    "Phim nhạt quá, {reason}",
                    "Diễn viên {acting_issue}, {problem}",
                    "Kịch bản {script_issue}, {details}",
                    "Thất vọng vì {aspect}, {reason}",
                    "Không đáng tiền vé, {details}",
                ],
                "neutral": [
                    "Phim cũng được, {reason}",
                    "Diễn viên {acting} tạm ổn, kịch bản {script}",
                    "Cốt truyện {plot_opinion}, {details}",
                    "Xem cũng được, {opinion}",
                    "Không quá tệ nhưng không xuất sắc, {details}"
                ],
            },
            "service_review": {
                "positive": [
                    "Nhân viên {staff_quality} và {staff_attitude}",
                    "Dịch vụ {service_quality}, {recommendation}",
                    "Được {good_point} và {another_point}",
                    "Rất {emotion} với {aspect}, {reason}",
                    "{service_type} ở đây {quality}, {details}",
                ],
                "negative": [
                    "Thái độ nhân viên {bad_attitude}, {issue}",
                    "Dịch vụ {service_issue}, {problem}",
                    "Không hài lòng về {aspect}, {reason}",
                    "Thất vọng về {issue_point}, {details}",
                    "{service_type} quá {negative_point}, {complaint}",
                ],
                "neutral": [
                    "Dịch vụ bình thường, {details}",
                    "Tạm được, nhưng {suggestion}",
                    "{aspect} có thể cải thiện thêm, {feedback}",
                    "Chưa có gì đặc sắc, {reason}",
                    "Cũng được, {opinion}",
                ],
            },
            "technology_review": {
                "positive": [
                    "{device} chạy {performance}, {details}",
                    "Cấu hình {spec_quality}, {feature_opinion}",
                    "Pin {battery_life}, {usage_experience}",
                    "Camera {camera_quality}, {photo_details}",
                    "Thiết kế {design_opinion}, {build_quality}",
                ],
                "negative": [
                    "{device} hay bị {tech_issue}, {problem}",
                    "Pin {battery_issue}, {complaint}",
                    "Giá quá {price_opinion} so với {comparison}",
                    "Cấu hình {spec_issue}, {performance_details}",
                    "Không đáng tiền vì {reason}, {details}",
                ],
                "neutral": [
                    "{device} dùng tạm được, {details}",
                    "Cấu hình {spec_quality} đủ dùng, {feature_opinion}",
                    "Pin {battery_life}, {usage_experience}",
                    "Camera {camera_quality}, {photo_details}",
                    "Thiết kế bình thường, {build_quality}"
                ],
            },
        }

        self.en_templates = {
            "product_review": {
                "positive": [
                    "This product is {quality}, {reason}",
                    "Really {emotion} with {aspect}",
                    "Great {packaging}, {delivery} shipping",
                    "{aspect} exceeded expectations, {details}",
                    "Price is {price_opinion}, {value_desc}",
                ],
                "negative": [
                    "{aspect} isn't good, {issue}",
                    "Disappointed with {aspect}, {reason}",
                    "Quality is {quality_issue}, {details}",
                    "Shipping was {delivery_issue}, {problem}",
                    "Not worth the money, {reason}",
                ],
            }
        }

        self.vi_fillers = {
            "quality": ["tốt", "xuất sắc", "tuyệt vời", "đỉnh", "chất lượng"],
            "emotion": ["hài lòng", "thích", "ưng", "mê", "yêu"],
            "packaging": ["cẩn thận", "chắc chắn", "đẹp", "gọn gàng"],
            "delivery": ["nhanh", "đúng hẹn", "tốt", "chuyên nghiệp"],
            "price_opinion": ["hợp lý", "rẻ", "tốt", "phải chăng"],
            "value_desc": ["đáng đồng tiền", "chất lượng xứng đáng", "rất hời"],
            "acting": ["diễn xuất tốt", "nhập vai", "tự nhiên", "thuyết phục"],
            "script": ["hay", "cuốn", "logic", "hấp dẫn"],
            "taste": ["ngon", "tuyệt", "xuất sắc", "đúng vị", "đậm đà"],
            "service": ["phục vụ", "tận tình", "nhiệt tình", "chuyên nghiệp"],
            "atmosphere": ["thoải mái", "đẹp", "sang trọng", "ấm cúng"],
            "staff_quality": [
                "chuyên nghiệp",
                "được đào tạo bài bản",
                "có kinh nghiệm",
            ],
            "staff_attitude": ["rất thân thiện", "nhiệt tình", "vui vẻ", "chu đáo"],
            "service_quality": ["rất tốt", "chuyên nghiệp", "đúng giờ", "nhanh chóng"],
            "good_point": [
                "tư vấn tận tình",
                "giải đáp thắc mắc rõ ràng",
                "hỗ trợ nhiệt tình",
            ],
            "bad_attitude": [
                "cọc cằn",
                "thiếu chuyên nghiệp",
                "không nhiệt tình",
                "làm việc qua loa",
            ],
            "service_issue": ["chậm trễ", "thiếu chuyên nghiệp", "không đúng cam kết"],
            "performance": ["mượt mà", "nhanh", "ổn định", "tốt", "lag"],
            "spec_quality": ["khá ổn", "mạnh mẽ", "đủ dùng", "cao cấp"],
            "battery_life": ["trâu", "tốt", "dùng được lâu", "không tốt"],
            "camera_quality": ["chụp đẹp", "chi tiết", "sắc nét", "tạm được"],
            "design_opinion": ["sang trọng", "đẹp", "hiện đại", "cao cấp"],
            "tech_issue": ["lag", "đơ", "nóng", "lỗi phần mềm"],
            "recommendation": [
                "nên thử nhé",
                "recommend mọi người nên dùng",
                "sẽ ủng hộ dài dài",
                "sẽ quay lại lần sau",
            ],
            "suggestion": [
                "cần cải thiện thêm",
                "có thể tốt hơn nữa",
                "nên nâng cấp dịch vụ",
            ],
            "details": [
                "thấy rất worth",
                "đáng đồng tiền",
                "giá hơi cao",
                "cần cải thiện thêm",
            ],
            "neutral_opinion": [
                "tạm được",
                "không có gì đặc biệt",
                "bình thường",
                "trung bình",
                "không nổi bật"
            ],
            "neutral_suggestion": [
                "có thể cải thiện thêm",
                "còn nhiều điểm cần phát triển",
                "cần nâng cấp thêm",
                "nên cải tiến"
            ],
            "neutral_aspect": [
                "chất lượng tạm ổn",
                "giá cả chấp nhận được",
                "dịch vụ bình thường",
                "không có gì để khen hoặc chê"
            ]
        }

        self.vi_aspects = {
            "product": ["chất lượng", "mẫu mã", "đóng gói", "giao hàng", "giá cả"],
            "food": ["hương vị", "phục vụ", "không gian", "giá cả", "vệ sinh"],
            "movie": ["nội dung", "diễn xuất", "kịch bản", "âm thanh", "hình ảnh"],
        }

        self.vi_expressions = {
            "positive": [
                "quá xịn luôn",
                "đỉnh thật sự",
                "không có gì để chê",
                "ưng cái bụng",
                "cực kỳ hài lòng",
                "xuất sắc",
            ],
            "negative": [
                "thất vọng quá",
                "chán thật sự",
                "không đáng tiền",
                "quá tệ",
                "không thể chấp nhận được",
            ],
            "neutral": [
                "tạm được",
                "không có gì đặc biệt",
                "bình thường",
                "cũng được",
                "tương đối ổn",
            ],
        }

        # Thêm từ lóng tiếng Việt
        self.vi_slangs = {
            "positive": {
                "xịn": ["xịn xò", "xịn sò", "đỉnh", "đỉnh cao", "cực phẩm"],
                "ngon": ["bá cháy", "bá đạo", "xuất sắc", "đỉnh của chóp"],
                "tốt": ["chất", "max good", "hết nước chấm", "không phải bàn"],
                "thích": ["khoái bá cháy", "ưng quá trời", "mê tít"],
                "hay": ["mãi đỉnh", "gút chóp", "max hay", "xịn sò"],
            },
            "negative": {
                "tệ": ["như cái bãi", "rác", "phèn", "dởm", "fail"],
                "kém": ["như hạch", "cùi bắp", "xác xơ"],
                "đắt": ["chát", "cắt cổ", "hút máu"],
                "chán": ["nhạt như nước ốc", "ngán ngẩm", "nản"],
                "dở": ["phế", "gà", "non", "trẻ trâu"],
            },
            "intensifiers": {
                "rất": ["đét", "quá xá", "dã man", "kinh hoàng"],
                "nhiều": ["ối dồi ôi", "vô số", "vô vàn"],
                "quá": ["vãi", "vl", "vcl", "xỉu up xỉu down"],
            },
            "internet_terms": {
                "ok": ["oce", "oke", "okela", "okê"],
                "không": ["kh", "hông", "khum", "hem"],
                "vậy": ["z", "dz", "v"],
                "được": ["đc", "dk", "dke"],
                "biết": ["bít", "bik", "bit"],
                "vui": ["zui", "zoui", "vkoj"],
                "buồn": ["bùn", "buon", "huhu"],
            },
            "neutral": {
                "bình thường": ["bt", "sương sương", "tàm tạm"],
                "tạm": ["tạm được", "được", "cũng được"],
                "trung bình": ["không đặc sắc", "không nổi bật"],
                "thường": ["bình bình", "không có gì đặc biệt"]
            },
            "informal_expressions": {
                "tức giận": ["tức ói", "điên tiết", "tức điên", "tức phát điên", "máu"],
                "thất vọng": ["chán đời", "nản vl", "chả buồn nói", "phát ngấy"],
                "phẫn nộ": ["đkm", "má nó", "dcm", "vkl", "ối dồi ôi"],
                "khen ngợi": ["đỉnh vl", "bá đạo vl", "max ngon", "xịn sò"],
                "chê bai": ["như c*t", "như sh*t", "như cức", "hãm vl", "tởm"],
                "bực mình": ["đ*o chịu nổi", "đ*o được", "quá mức chịu đựng"],
                "bất ngờ": ["đậu má", "vãi cả l*n", "vãi", "vcl"]
            }
        }

        # Add more slang variations to existing categories
        self.vi_slangs["positive"].update({
            "xịn": self.vi_slangs["positive"]["xịn"] + ["đỉnh vl", "xịn sò vl"],
            "ngon": self.vi_slangs["positive"]["ngon"] + ["ngon vl", "đỉnh của chóp vl"],
            "tốt": self.vi_slangs["positive"]["tốt"] + ["quá mẹ ngon", "đỉnh quá xá"],
            "thích": self.vi_slangs["positive"]["thích"] + ["phê vl", "sướng phát xỉu"]
        })

        self.vi_slangs["negative"].update({
            "tệ": self.vi_slangs["negative"]["tệ"] + ["như c*t", "như sh*t"],
            "kém": self.vi_slangs["negative"]["kém"] + ["như hạch vl", "dở ẹc"],
            "đắt": self.vi_slangs["negative"]["đắt"] + ["chém gió vl", "cướp tiền"],
            "dở": self.vi_slangs["negative"]["dở"] + ["ngu vl", "gà vl"]
        })

        # Add more internet terms
        self.vi_slangs["internet_terms"].update({
            "không": self.vi_slangs["internet_terms"]["không"] + ["éo", "đ*o", "đéo"],
            "vãi": ["v~", "vl", "vcl", "vloz"],
            "quá": ["vãi cả l", "vcl", "vl"],
            "được": ["đc", "dk", "được của l*"]
        })

        # Thêm từ lóng tiếng Anh
        self.en_slangs = {
            "positive": {
                "good": ["lit", "fire", "dope", "sick", "rad"],
                "great": ["goated", "bussin", "slaps", "hits different"],
                "amazing": ["baddie", "based", "poggers", "absolute unit"],
                "like": ["stan", "vibe with", "fuck with", "dig"],
                "perfect": ["no cap", "straight fire", "hits hard"],
            },
            "negative": {
                "bad": ["mid", "trash", "cap", "sus", "ain't it"],
                "terrible": ["wack", "garbage", "dead", "basic"],
                "expensive": ["pricey af", "costs a bag", "steep"],
                "boring": ["sleeping on it", "dry", "dead"],
                "fake": ["cap", "sus", "fugazi", "bogus"],
            },
            "intensifiers": {
                "very": ["af", "asf", "fr fr", "ong"],
                "really": ["deadass", "fr", "no cap", "straight up"],
                "absolutely": ["lowkey", "highkey", "straight up"],
            },
            "internet_terms": {
                "okay": ["k", "kk", "aight", "ight"],
                "thanks": ["ty", "thx", "thnx"],
                "please": ["pls", "plz", "plox"],
                "what": ["wat", "wut", "tf"],
                "lol": ["lmao", "lmfao", "rofl"],
                "omg": ["omfg", "bruh", "bruhhh"],
            },
            "neutral": {
                "okay": ["meh", "whatever", "so-so"],
                "average": ["decent", "fair", "normal"],
                "mediocre": ["basic", "standard", "regular"],
                "moderate": ["alright", "passable", "fine"]
            }
        }

        # Thêm template mới sử dụng từ lóng
        self.vi_templates["social_media_review"] = {
            "positive": [
                "Ẩm thực {location} {slang_positive} luôn, {intensifier} {good_point}",
                "Quán này {slang_positive} {intensifier}, {recommendation}",
                "Giá hơi chát nhưng mà {slang_positive} thật, {details}",
                "{aspect} thì {slang_positive} khỏi bàn, {intensifier} {opinion}",
                "Nhân viên {staff_quality} {intensifier}, {staff_attitude}",
            ],
            "negative": [
                "Đúng là {slang_negative} thật sự, {intensifier} {issue}",
                "Quán này {slang_negative} {intensifier}, {problem}",
                "Giá thì {price_complaint} mà {slang_negative}, {details}",
                "{aspect} thì {slang_negative} {intensifier}, {complaint}",
                "Thái độ nhân viên {bad_attitude}, {slang_negative} {intensifier}",
            ],
        }

    def get_random_slang(
        self, sentiment: str, category: str, language: str = "vi"
    ) -> str:
        """Get random slang based on sentiment and category"""
        slang_dict = self.vi_slangs if language == "vi" else self.en_slangs
        if category in slang_dict and sentiment in slang_dict[category]:
            return random.choice(slang_dict[category][sentiment])
        return ""

    def get_random_intensifier(self, language: str = "vi") -> str:
        """Get random intensifier"""
        slang_dict = self.vi_slangs if language == "vi" else self.en_slangs
        return random.choice(
            slang_dict["intensifiers"]["rất" if language == "vi" else "very"]
        )

    def get_internet_term(self, word: str, language: str = "vi") -> str:
        """Get internet slang version of a word if available"""
        slang_dict = self.vi_slangs if language == "vi" else self.en_slangs
        if word.lower() in slang_dict["internet_terms"]:
            return random.choice(slang_dict["internet_terms"][word.lower()])
        return word
