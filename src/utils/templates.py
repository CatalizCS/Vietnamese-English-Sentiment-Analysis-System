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
                "neutral": [
                    "Product is okay, {details}",
                    "Decent but {suggestion}",
                    "Nothing special, {reason}",
                    "Usable product, {opinion}",
                    "Fairly standard, {details}"
                ]
            }
        }

        self.en_templates.update({
            "food_review": {
                "positive": [
                    "Food is {taste}, {details}",
                    "Service is {service_quality}, {atmosphere}",
                    "Quality is {quality}, {reason}",
                    "Great value for {price_opinion}, {value}",
                    "Ambiance is {atmosphere}, {recommendation}"
                ],
                "negative": [
                    "Food was {taste_issue}, {reason}",
                    "Poor service: {service_issue}, {problem}",
                    "Overpriced: {price_complaint}, {details}",
                    "Hygiene issues: {cleanliness_issue}, {problem}",
                    "Disappointed with {aspect}, {reason}"
                ],
                "neutral": [
                    "The food is average, {details}",
                    "Price is reasonable, {value}",
                    "Service is decent, {details}",
                    "Ambiance is okay but {suggestion}",
                    "Standard quality, {opinion}"
                ]
            },
            "movie_review": {
                "positive": [
                    "Great movie, {reason}",
                    "Actors were {acting}, script was {script}",
                    "Plot was {plot_opinion}, {details}",
                    "Felt {emotion} watching it, {reason}",
                    "Worth watching, {recommendation}"
                ],
                "negative": [
                    "Boring movie, {reason}",
                    "Actors were {acting_issue}, {problem}",
                    "Script was {script_issue}, {details}",
                    "Disappointed with {aspect}, {reason}",
                    "Not worth the ticket price, {details}"
                ],
                "neutral": [
                    "Movie is alright, {reason}",
                    "Acting is decent, script is {script}",
                    "Plot is {plot_opinion}, {details}",
                    "It's watchable, {opinion}",
                    "Neither great nor terrible, {details}"
                ]
            },
            "service_review": {
                "positive": [
                    "Staff were {staff_quality} and {staff_attitude}",
                    "Service was {service_quality}, {recommendation}",
                    "Had {good_point} and {another_point}",
                    "Very {emotion} with {aspect}, {reason}",
                    "{service_type} here is {quality}, {details}"
                ],
                "negative": [
                    "Staff attitude was {bad_attitude}, {issue}",
                    "Service was {service_issue}, {problem}",
                    "Not satisfied with {aspect}, {reason}",
                    "Disappointed with {issue_point}, {details}",
                    "{service_type} was too {negative_point}, {complaint}"
                ],
                "neutral": [
                    "Service was okay, {details}",
                    "Decent but {suggestion}",
                    "{aspect} could be improved, {feedback}",
                    "Nothing remarkable, {reason}",
                    "Standard service, {opinion}"
                ]
            }
        })

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

        self.vi_fillers.update({
            "quality": [
                "tốt", "xuất sắc", "tuyệt vời", "đỉnh", "chất lượng",
                "xịn xò", "đáng đồng tiền", "ưng cái bụng", "ngon lành", 
                "không phải dạng vừa đâu"
            ],
            "emotion": [
                "hài lòng", "thích", "ưng", "mê", "yêu",
                "phê quá trời", "sướng rớt nước miếng", "đê mê", 
                "mê tít thò lò", "phải lòng ngay cái nhìn đầu tiên"
            ],
        })

        self.en_fillers = {
            "quality": ["good", "excellent", "amazing", "great", "outstanding"],
            "emotion": ["happy", "satisfied", "pleased", "delighted", "impressed"],
            "packaging": ["careful", "secure", "nice", "neat", "professional"],
            "delivery": ["fast", "on-time", "efficient", "professional", "prompt"],
            "price_opinion": ["reasonable", "affordable", "fair", "good value"],
            "value_desc": ["worth the money", "great value", "excellent deal"],
            "reason": ["really love the quality", "exceeds expectations", "exactly what I needed"],
            "issue": ["not up to standard", "below expectations", "poor quality"],
            "problem": ["constant issues", "major flaws", "serious problems"],
            "details": ["highly recommend", "would buy again", "excellent purchase"],
            "suggestion": ["could be improved", "needs work", "should be better"],
            "recommendation": ["definitely recommend", "worth trying", "must buy"],
            "quality_issue": ["poor", "substandard", "disappointing", "terrible"],
            "delivery_issue": ["delayed", "late", "unprofessional", "problematic"],
            "aspect": ["quality", "design", "functionality", "performance", "features"],
            "neutral_opinion": [
                "decent", "average", "okay", "standard",
                "fair", "moderate", "acceptable", "passable"
            ],
            "neutral_suggestion": [
                "could be improved",
                "needs some work",
                "has room for improvement",
                "could be better"
            ],
            "neutral_quality": [
                "average quality",
                "acceptable standard",
                "middle-ground",
                "fair enough"
            ],
            "neutral_response": [
                "mixed feelings",
                "balanced view",
                "moderate opinion",
                "neutral stance"
            ]
        }

        missing_en_fillers = {
            "taste": ["delicious", "amazing", "excellent", "flavorful", "tasty"],
            "service": ["attentive", "professional", "friendly", "efficient"],
            "atmosphere": ["cozy", "elegant", "comfortable", "pleasant"],
            "staff_quality": ["well-trained", "experienced", "professional", "skilled"],
            "staff_attitude": ["friendly", "helpful", "courteous", "welcoming"],
            # ...add other missing categories
        }
        
        self.en_fillers.update(missing_en_fillers)

        self.vi_aspects = {
            "product": ["chất lượng", "mẫu mã", "đóng gói", "giao hàng", "giá cả"],
            "food": ["hương vị", "phục vụ", "không gian", "giá cả", "vệ sinh"],
            "movie": ["nội dung", "diễn xuất", "kịch bản", "âm thanh", "hình ảnh"],
        }

        self.en_aspects = {
            "product": ["quality", "design", "packaging", "shipping", "price"],
            "food": ["taste", "service", "ambiance", "pricing", "cleanliness"],
            "movie": ["content", "acting", "script", "sound", "visuals"],
            "service": ["staff", "efficiency", "value", "facilities", "experience"]
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

        # Thêm từ điển emoji phù hợp với cảm xúc
        self.emojis = {
            "positive": ["😊", "🥰", "😍", "🤩", "👍", "💯", "🔥", "✨", "💪", "����"],
            "negative": ["😤", "😒", "😑", "👎", "😠", "😡", "🤬", "💢", "😫", "😩"],
            "neutral": ["🤔", "😐", "😶", "🤷", "😕", "😌", "🙂", "👀"]
        }

        # Thêm các cách diễn đạt tự nhiên
        self.natural_expressions = {
            "opening": [
                "Thật sự thì", "Nói thật là", "Theo mình thấy thì", 
                "Mình đánh giá là", "Cá nhân mình thấy",
                "Xin chia sẻ chút là", "Mình mới dùng thử và thấy",
                "Sau thời gian trải nghiệm thì"
            ],
            "closing": [
                "Đó là góc nhìn của mình ạ", "Mọi người thấy sao?",
                "Đánh giá chủ quan thôi nha", "Hy vọng review có ích",
                "Mình nghĩ vậy thôi", "Mọi người cân nhắc nha"
            ]
        }

        # Thêm mẫu câu ngắn
        self.short_templates = {
            "positive": [
                "Tuyệt vời 👍",
                "Quá ngon luôn",
                "Đỉnh thật sự",
                "Rất ưng ý",
                "Xứng đáng {score}/10",
                "Không có gì để chê",
                "Sẽ quay lại lần sau",
                "Recommend nha mọi người"
            ],
            "negative": [
                "Thất vọng quá",
                "Không đáng tiền",
                "Chán thật sự",
                "Tệ hết chỗ nói",
                "Không bao giờ quay lại",
                "Phí tiền",
                "Quá tệ {score}/10",
                "Không nên mua"
            ],
            "neutral": [
                "Tạm được",
                "Bình thường",
                "Cũng được",
                "Không có gì đặc biệt",
                "Tương đối ổn",
                "{score}/10 thôi",
                "Còn cải thiện được"
            ]
        }

        # Thêm mẫu câu dài, chi tiết
        self.long_templates = {
            "positive": [
                "Đây là lần thứ {count} mình {action} và vẫn rất {emotion}. {aspect} thì {quality}, đặc biệt là {highlight}. {recommendation}",
                "Thực sự {emotion} khi {action}. {details} Về {aspect} thì {quality}, còn {another_aspect} cũng {another_quality}. {conclusion}",
                "Mình đã {action} được {duration} rồi và phải nói là {quality}. {reason} Ngoài ra {additional_point}. {suggestion}",
                "Trải nghiệm {duration} với {product/service} này thì thấy {overall_feeling}. {aspect} thì {quality}, {another_aspect} thì {another_quality}. {detailed_review} {final_thought}"
            ],
            "negative": [
                "Thất vọng tột độ với {aspect}. {issue_details} Không những thế, {another_issue}. {complaint_details} Mình đã liên hệ {support_channel} nhưng {service_issue}. {conclusion}",
                "Đây là trải nghiệm tệ nhất từ trước đến nay với {product/service}. {main_issue} Thêm vào đó, {additional_issues}. {negative_impact} {warning}",
                "Mình đã cho cơ hội {count} lần nhưng {recurring_issue}. {details} Về {aspect} thì {quality_issue}, {service_complaint}. {final_warning}"
            ],
            "neutral": [
                "Sau {duration} sử dụng thì thấy {product/service} này {neutral_opinion}. {positive_points} nhưng {negative_points}. {improvement_suggestions}",
                "Không quá tệ nhưng cũng không xuất sắc. {aspect} thì {neutral_quality}, còn {another_aspect} thì {areas_for_improvement}. {balanced_conclusion}",
                "Mình thấy {product/service} này còn nhiều điểm cần cải thiện. {details} Tuy nhiên cũng có {positive_aspects}. {suggestions}"
            ]
        }

        # Thêm từ điển điểm số và thời lượng
        self.scores = {
            "positive": ["9", "9.5", "10", "8.5"],
            "negative": ["2", "3", "4", "1"],
            "neutral": ["5", "6", "7", "6.5"]
        }
        
        self.durations = [
            "một thời gian", "mấy tháng", "gần năm", 
            "một tuần", "vài ngày", "khá lâu",
            "hơn {number} tháng", "gần {number} năm",
            "được {number} lần"
        ]

        self.numbers = ["1", "2", "3", "4", "5", "nhiều"]

        self.en_ratings = {
            "positive": [
                "{score}/10 would recommend",
                "Solid {score}/10",
                "A strong {score} out of 10",
                "Definitely {score}/10"
            ],
            "negative": [
                "Unfortunately {score}/10",
                "Disappointing {score}/10",
                "A weak {score}/10",
                "Only {score}/10"
            ],
            "neutral": [
                "Average {score}/10",
                "Middle-of-the-road {score}/10",
                "Fair {score}/10",
                "Decent {score}/10"
            ]
        }

        self.en_durations = [
            "for a while", "for months", "nearly a year",
            "for a week", "several days", "quite some time",
            "over {number} months", "almost {number} years",
            "{number} times"
        ]

        # Add English expressions
        self.en_expressions = {
            "positive": [
                "absolutely fantastic",
                "really amazing",
                "nothing to complain about",
                "exceeded expectations",
                "extremely satisfied",
                "outstanding",
                "top notch",
                "brilliant"
            ],
            "negative": [
                "very disappointing",
                "really frustrating",
                "not worth the money",
                "terrible experience",
                "completely unacceptable",
                "worst ever",
                "absolute garbage"
            ],
            "neutral": [
                "fairly decent",
                "nothing special",
                "average",
                "okay",
                "relatively fine",
                "mediocre",
                "standard"
            ]
        }

        # Add English natural expressions
        self.en_natural_expressions = {
            "opening": [
                "To be honest,",
                "In my experience,",
                "From my perspective,",
                "After trying this,",
                "I have to say,",
                "Let me share that",
                "Based on my usage,",
                "After some time with this,"
            ],
            "closing": [
                "That's my take on it.",
                "What do you think?",
                "Just my personal opinion.",
                "Hope this helps!",
                "That's my perspective.",
                "Consider it before buying."
            ]
        }

        # Thêm mẫu bình luận tương tác
        self.interaction_templates = {
            "argument": {
                "aggressive": [
                    "Đ* biết gì mà {action}? {insult}",
                    "Ngu như {insult} mà cũng {action}",
                    "M là thằng {insult} à? Sao {action} v?",
                    "Làm như hay lắm í, {insult}",
                    "Ăn nói như {insult}, blocked!",
                ],
                "defensive": [
                    "Ai cho {subject} {action}? {counter_argument}",
                    "Mắc gì phải nghe {subject}? {dismissal}",
                    "Kệ tao, liên quan gì đến {subject}?",
                    "Đừng có mà {action}, {warning}",
                    "Nói nữa là {threat} đấy!",
                ],
                "dismissive": [
                    "Kệ người ta đi {subject} ơi",
                    "Thôi {subject} ạ, chả ai quan tâm đâu",
                    "Đ* ai thèm để ý {subject} nói gì",
                    "Lạ nhỉ? Ai hỏi {subject} không?",
                    "Không ai cần ý kiến của {subject} đâu",
                ]
            },
            "support": {
                "agreement": [
                    "+1 với {subject}, {reason}",
                    "Đồng ý với {subject} luôn, {explanation}",
                    "Chuẩn đấy {subject} ơi! {detail}",
                    "Như {subject} nói là đúng rồi",
                    "{subject} nói chuẩn quá! {agreement}"
                ],
                "praise": [
                    "Review hay quá {subject} ơi! {appreciation}",
                    "Tks {subject} đã chia sẻ nha! {gratitude}",
                    "Bài viết chất lượng {subject} ạ",
                    "Góp ý có tâm quá {subject}",
                    "Respect {subject}! {reason}"
                ]
            },
            "trolling": [
                "Ơ thế {subject} định nói gì? 🤡",
                "Nghe {subject} nói mà tức cười quá 😂",
                "Thím {subject} lại nổi hứng rồi",
                "Cao thủ {subject} lại xuất hiện kìa ae 🤣",
                "Đọc mà xỉu với {subject} luôn"
            ]
        }

        # Thêm từ vựng cho bình luận tương tác
        self.interaction_fillers = {
            "insult": [
                "ngu như bò", "óc chó", "đầu đất", "thiểu năng", 
                "ăn c*t", "ngu l*n", "mặt người óc lợn",
                "đần độn", "ngu si", "ngáo đá"
            ],
            "action": [
                "bô bô cái mồm", "phát biểu", "lên mặt dạy đời",
                "sủa", "gâu gâu", "hùa theo", "nói năng lung tung",
                "xàm xí", "thể hiện", "xen vào"
            ],
            "subject": [
                "bạn", "thím", "bác", "chế", "đồng chí",
                "cụ", "anh", "chị", "bợn", "đại ca"
            ],
            "threat": [
                "ăn block", "báo admin", "cho lên thớt",
                "cho ra đảo", "tay không bắt giặc", 
                "xử đẹp", "đập phát chết luôn"
            ],
            "counter_argument": [
                "nói cho biết nha", "nhớ đấy nhá",
                "cãi là ăn ban", "đừng có mà ngồi mơ",
                "tự soi gương đi"
            ],
            "dismissal": [
                "lo chuyện của mình đi", "đi ngủ sớm đi",
                "về mà hỏi google", "dứt ra cho nước nó trong",
                "lượn đi cho nước nó trong"
            ],
            "warning": [
                "đừng để tôi nóng", "cẩn thận cái mồm",
                "liệu mà giữ mồm", "coi chừng tay tôi",
                "cẩn thận kẻo đấm nhau"
            ],
            "appreciation": [
                "review có tâm quá", "chia sẻ xịn xò",
                "góp ý quá chuẩn", "phân tích rất hay",
                "đánh giá rất khách quan"
            ]
        }

        clean_vi_interaction = {
            "insult": [
                "không hiểu gì", "thiếu hiểu biết", "kém cỏi",
                "không có kiến thức", "thiếu kinh nghiệm"
            ],
            "action": [
                "lên tiếng", "phát biểu", "góp ý",
                "bình luận", "đánh giá", "phê bình"
            ],
            "warning": [
                "cẩn thận lời nói", "giữ ý một chút",
                "suy nghĩ kỹ hơn", "điều chỉnh cách nói",
                "xem lại cách ứng xử"
            ]
        }
        
        # Replace offensive terms with clean ones
        self.interaction_fillers.update(clean_vi_interaction)

        # Add English interaction templates
        self.en_interaction_templates = {
            "argument": {
                "aggressive": [
                    "What do you know about {action}?",
                    "You clearly don't understand {topic}",
                    "Stop talking nonsense about {topic}",
                    "Your opinion is totally wrong",
                    "You have no idea what you're saying"
                ],
                "defensive": [
                    "Who asked for your opinion?",
                    "Mind your own business",
                    "Whatever, I don't care what you think",
                    "Don't tell me what to do",
                    "You should know better"
                ],
                "dismissive": [
                    "Just ignore them",
                    "Nobody cares about that opinion",
                    "Why even bother responding?",
                    "Not worth discussing",
                    "Let's move on from this"
                ]
            },
            "support": {
                "agreement": [
                    "Totally agree with you about {topic}!",
                    "You're absolutely right, {reason}",
                    "Couldn't agree more! {detail}",
                    "That's exactly what I think",
                    "Well said! {agreement}"
                ],
                "praise": [
                    "Great review! {appreciation}",
                    "Thanks for sharing! {gratitude}",
                    "Very helpful review",
                    "Excellent feedback",
                    "Really appreciate your insights!"
                ]
            },
            "neutral": {
                "balanced": [
                    "I see both sides here, {topic}",
                    "There are pros and cons, {details}",
                    "It's not that simple, {explanation}",
                    "Let's be objective about {topic}",
                    "Consider both perspectives on {subject}"
                ],
                "moderate": [
                    "Maybe we should wait and see",
                    "Not jumping to conclusions about {topic}",
                    "Need more information about {subject}",
                    "Taking a balanced view on this",
                    "Looking at it objectively"
                ]
            }
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

    def get_random_duration(self):
        """Get random duration with optional number"""
        duration = random.choice(self.durations)
        if "{number}" in duration:
            duration = duration.replace("{number}", random.choice(self.numbers))
        return duration

    def get_random_score(self, sentiment):
        """Get appropriate score based on sentiment"""
        return random.choice(self.scores[sentiment])

    def generate_varied_length_comment(self, sentiment, topic):
        """Generate comment with varied length"""
        if random.random() < 0.3:  # 30% chance for short comment
            template = random.choice(self.short_templates[sentiment])
            if "{score}" in template:
                template = template.replace("{score}", self.get_random_score(sentiment))
            return template
        elif random.random() < 0.7:  # 40% chance for normal comment
            return self.generate_normal_comment(sentiment, topic)
        else:  # 30% chance for long comment
            template = random.choice(self.long_templates[sentiment])
            return self.fill_long_template(template, sentiment, topic)

    def fill_long_template(self, template, sentiment, topic):
        """Fill in the placeholders for long templates with appropriate content"""
        replacements = {
            "{count}": random.choice(self.numbers),
            "{duration}": self.get_random_duration(),
            "{action}": self.get_random_action(topic),
            "{emotion}": self.get_random_emotion(sentiment),
        }
        
        for key, value in replacements.items():
            if key in template:
                template = template.replace(key, value)
        return template

    def get_random_action(self, topic: str, language: str = "vi") -> str:
        """Get random action based on topic and language"""
        actions = {
            "vi": {
                "general": ["dùng", "sử dụng", "trải nghiệm", "mua"],
                "food": ["ăn", "thử", "ghé quán", "đặt đồ"],
                "movie": ["xem", "ra rạp xem", "thưởng thức"],
                "service": ["sử dụng dịch vụ", "trải nghiệm", "thuê"]
            },
            "en": {
                "general": ["used", "tried", "experienced", "purchased"],
                "food": ["ate at", "tried", "visited", "ordered from"],
                "movie": ["watched", "saw", "experienced"],
                "service": ["used the service", "experienced", "hired"]
            }
        }
        
        topic_actions = actions[language].get(topic, actions[language]["general"])
        return random.choice(topic_actions)

    def get_random_emotion(self, sentiment: str, language: str = "vi") -> str:
        """Get random emotion based on sentiment and language"""
        emotions = {
            "vi": {
                "positive": self.vi_fillers["emotion"],
                "negative": ["thất vọng", "buồn", "không hài lòng", "bực mình"],
                "neutral": ["bình thường", "tạm được", "không đặc biệt"]
            },
            "en": {
                "positive": self.en_fillers["emotion"],
                "negative": ["disappointed", "upset", "dissatisfied", "frustrated"],
                "neutral": ["okay", "alright", "not special", "decent"]
            }
        }
        
        return random.choice(emotions[language][sentiment])

    def generate_interaction_comment(self, interaction_type: str, sub_type: str = None) -> str:
        """Generate an interaction comment"""
        if sub_type:
            template = random.choice(self.interaction_templates[interaction_type][sub_type])
        else:
            template = random.choice(self.interaction_templates[interaction_type])

        # Fill template with random fillers
        for key, values in self.interaction_fillers.items():
            if "{" + key + "}" in template:
                template = template.replace("{" + key + "}", random.choice(values))

        return template
