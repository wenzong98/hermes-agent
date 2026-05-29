#!/usr/bin/env python3
"""
Blogger Knowledge Base & Semantic Matcher
结合博主知识库、分词和语义的精细化博主识别系统

核心改进:
1. 加载博主知识库，构建别名映射表
2. 基于知识库的分词：识别出已知的博主名token
3. 语义匹配：平台+标签+风格的语义关联
4. 别名归一化：同一博主的不同写法统一映射
5. 置信度评分：每个匹配结果带置信度
"""
import os
import re
import json
import unicodedata
from collections import defaultdict
from datetime import datetime


BASE_PATH = "/Volumes/personal_folder-1/movie_a/中文"
KB_PATH = "/Users/bytedance/.hermes/notes/博主知识库.md"
ALIAS_DB_PATH = "/Users/bytedance/.hermes/hermes-agent/blogger_aliases.json"


PLATFORM_MAP = {
    "推特": ["twitter", "推特", "tw", "x.com"],
    "Twitter": ["twitter", "推特", "tw", "x.com"],
    "Myfans": ["myfans", "mf", "myfans/candfans"],
    "Candfans": ["candfans", "cf", "candfans/myfans"],
    "OnlyFans": ["onlyfans", "of", "of_", "onlyfans_"],
    "Fansly": ["fansly", "fl"],
    "PikPak": ["pikpak", "pk", "p盘"],
    "P站": ["pornhub", "p站", "ph"],
    "91大神": ["91", "91大神", "91porn"],
    "FansOne": ["fansone", "fans1"],
    "Fantia": ["fantia", "ft"],
    "Stripchat": ["stripchat", "st站", "st", "cb站"],
    "抖音": ["抖音", "douyin", "dy"],
    "快手": ["快手", "kuaishou", "ks"],
    "B站": ["b站", "bilibili", "哔哩哔哩"],
    "微博": ["微博", "weibo", "wb"],
    "小红书": ["小红书", "xiaohongshu", "xhs"],
    "国产": ["国产", "cn", "chinese"],
    "韩国": ["韩国", "korean", "kr", "korea"],
    "日本": ["日本", "japanese", "jp", "japan"],
    "欧美": ["欧美", "western", "eu"],
    "FC2": ["fc2", "fc2-ppv"],
}

TAG_SEMANTIC_GROUPS = {
    "巨乳系": ["巨乳", "大奶", "大胸", "爆乳", "G罩杯", "H罩杯", "K罩杯", "N罩杯",
              "F罩杯", "奶", "乳", "美乳", "大奶子", "奶子", "乳交"],
    "约炮系": ["约炮", "约啪", "泡良", "良家", "搭讪", "真人", "实拍"],
    "反差系": ["反差", "反差婊", "反差母狗", "清纯反差"],
    "露出系": ["露出", "户外", "全裸", "野战", "路边"],
    "调教系": ["调教", "SM", "dom", "抖M", "母狗", "性奴", "肉便器", "榨精",
              "足交", "FEMDOM", "控精"],
    "NTR系": ["NTR", "绿帽", "淫妻", "寝取", "绿妻", " cuckold"],
    "COS系": ["cos", "cosplay", "coser", "角色", "原神", "汉服"],
    "ASMR系": ["asmr", "声优", "舔耳"],
    "人妻系": ["人妻", "少妇", "已婚", "熟女"],
    "学生系": ["学生", "学妹", "校花", "大学生", "jk", "女大"],
    "网红系": ["网红", "网黄", "博主", "主播", "福利姬"],
    "体育生系": ["体育生", "肌肉", "猛男", "健身", "dom"],
}

BLOGGER_ALIASES = {
    "黑杰克": ["黑杰克", "黑傑克", "heijieke"],
    "fkabuto": ["fkabuto", "fKabuto", "FKabuto", "fkabuto02", "fKabutoX"],
    "ROSEWU": ["ROSEWU", "rosewu", "RoseWu"],
    "ICICIS": ["ICICIS", "icicis", "Icicis"],
    "mini咪妮": ["mini咪妮", "mini米妮", "minimini"],
    "nana_taipei": ["nana_taipei", "娜娜", "nana", "nanataipei"],
    "xoxo_yuri": ["xoxo_yuri", "yuri", "Yuri", "onlyfans_xoxo_yuri"],
    "luckydog7": ["luckydog7", "LuckyDog7", "LuckyDog77", "luckydog"],
    "Aheyanlz": ["Aheyanlz", "aheyanlz"],
    "Enthus1asmm": ["Enthus1asmm", "enthus1asmm", "enthusiasmm"],
    "深海杀人鲸_小张历险记": ["深海杀人鲸", "小张历险记", "深海杀人鲸_小张历险记", "Ubersexx24h"],
    "噗噗通通": ["噗噗通通", "pupuv1", "pupuwaifu"],
    "YoShiE冰块": ["YoShiE冰块", "yoshie", "Yoshie冰块"],
    "香菜老师": ["香菜老师", "xiangcaiking1", "香菜", "xiangcai"],
    "sivr00394": ["sivr00394"],
    "烈": ["烈", "retsu_dao", "Retsu_dao", "retsu"],
    "muchi_tina": ["muchi_tina", "muchitinasub", "muchi_tina/ティナ"],
    "黑饱宝": ["黑饱宝", "黑闰润", "黑饱宝/黑闰润"],
    "白日梦想鸭": ["白日梦想鸭", "YAYA", "yaya"],
    "91大庆哥": ["91大庆哥", "大庆哥"],
    "91Tims": ["91Tims", "91tims"],
    "91Mrber": ["91Mrber", "91Mrber泰迪"],
    "newyearst6": ["newyearst6", "newyearst"],
    "跳跳羊": ["跳跳羊"],
    "小杏仁": ["小杏仁"],
    "好大的爆米花": ["好大的爆米花", "爆米花"],
    "古河君": ["古河君"],
    "dulianmaomao": ["dulianmaomao"],
    "slamdrunk": ["slamdrunk"],
    "Bigfan13yo": ["Bigfan13yo", "bigfan13yo"],
    "Chloe霏霏": ["Chloe霏霏", "chloe霏霏", "霏霏"],
    "鱼哥": ["鱼哥", "霸王别姬"],
    "韩国大叔": ["韩国大叔"],
    "西萌": ["西萌", "濑濑", "西萌工作室"],
    "柚子猫": ["柚子猫"],
    "铃木美咲": ["铃木美咲", "美咲", "Misaki"],
    "苏畅": ["苏畅"],
    "兔子先生": ["兔子先生", "TZ"],
    "kuzu_v0": ["kuzu_v0", "kuzu", "kuzuv0"],
    "shinosuke": ["shinosuke", "しんのすけ"],
    "sleepyboy_x": ["sleepyboy_x", "sleepyboy"],
    "072q": ["072q"],
    "retsu_dao": ["retsu_dao", "烈", "Retsu_dao"],
    "velevt8800": ["velevt8800", "velvet8800"],
    "FENDSON": ["FENDSON", "fendson"],
    "苹果": ["苹果", "Apple Creampie", "AppleCreampie"],
    "海绵宝宝": ["海绵宝宝"],
    "武汉情侣": ["武汉情侣"],
    "困困狗": ["困困狗"],
    "一个ren": ["一个ren", "一个人"],
    "JOJO杰克": ["JOJO杰克", "jaacckk999235", "jjaacckkyy72727"],
    "隔壁王某某": ["隔壁王某某"],
    "采精的小蝴蝶": ["采精的小蝴蝶"],
    "狗爹和小桃": ["狗爹和小桃"],
    "饭饭吖": ["饭饭吖"],
    "恩凯Enkai": ["恩凯Enkai", "Enkai", "恩凯"],
    "上官太太": ["上官太太"],
    "梓怡学妹": ["梓怡学妹"],
    "菠萝啤": ["菠萝啤"],
    "Coco": ["Coco", "Cocopie", "cocopie"],
    "慧慧": ["慧慧"],
    "楠熙": ["楠熙"],
    "小千": ["小千"],
    "小浅": ["小浅", "小浅同学"],
    "林淑": ["林淑", "LSY856", "林淑怡"],
    "蜜桃淳": ["蜜桃淳", "蜜桃"],
    "大屁股猪猪": ["大屁股猪猪"],
    "叶子姐姐": ["叶子姐姐", "木兰户外", "叶娇娇"],
    "白桃汽水": ["白桃汽水"],
    "警兔": ["警兔"],
    "鹅美美": ["鹅美美"],
    "辉夜姬": ["辉夜姬", "玉汇", "Kokuhui"],
    "爱吃雪糕": ["爱吃雪糕"],
    "高端调教": ["高端调教"],
    "午夜玫瑰喵喵": ["午夜玫瑰喵喵"],
    "苏酥学姐": ["苏酥学姐"],
    "小灰灰呐": ["小灰灰呐"],
    "你的宇吖": ["你的宇吖"],
    "Hahaha_ha2": ["Hahaha_ha2"],
    "raikun325": ["raikun325", "五条ライ"],
    "minato___26": ["minato___26", "みなと", "minato", "水原聖子"],
    "vzrym9": ["vzrym9", "ジェイ"],
    "nodoboko": ["nodoboko", "虎丸"],
    "Banbi_555": ["Banbi_555", "Banbi_555J"],
    "secret_japan": ["secret_japan", "裏垢Japan"],
    "Qastaado": ["Qastaado"],
    "ycancan": ["ycancan"],
    "kaori_xoxo": ["kaori_xoxo", "かおり"],
    "Ruri_LapisL": ["Ruri_LapisL", "ルリ"],
    "kuramakun": ["kuramakun", "くらがく"],
    "komachi": ["komachi", "小丁komachi"],
    "eroman": ["eroman"],
    "kemonokai": ["kemonokai", "獣会"],
    "Cyndaquil_log": ["Cyndaquil_log"],
    "suita_ona_ka": ["suita_ona_ka", "つばさ"],
    "rinrin_meow": ["rinrin_meow"],
    "ray_sama": ["ray_sama"],
    "jerry_jun": ["jerry_jun"],
    "toyboysamoari": ["toyboysamoari"],
    "Piston mafia": ["Piston mafia"],
    "yuma24": ["yuma24", "灯葉", "あきは"],
    "tokyogirlscatalog": ["tokyogirlscatalog"],
    "Momoeri.angel": ["Momoeri.angel"],
    "072q": ["072q"],
    "YuiPiSM": ["YuiPiSM", "雌犬ゆい"],
    "shuju": ["shuju"],
    "deep666": ["deep666", "六六男神"],
    "小Q小K": ["小Q小K", "qqq_qq77"],
    "HenTaipei": ["HenTaipei"],
    "943162807": ["943162807"],
    "91Xxyy09876": ["91Xxyy09876"],
    "__rose001": ["__rose001", "ロゼ"],
    "RockTangg": ["RockTangg"],
    "77bandage": ["77bandage"],
    "luobao1221": ["luobao1221"],
    "九儿": ["九儿", "wrmm520"],
    "姐妹": ["姐妹", "33jiemm"],
    "TT婉婉晴晴哒": ["TT婉婉晴晴哒"],
    "yan_3077": ["yan_3077", "艾嫣"],
    "sophiekimm": ["sophiekimm"],
    "纲手": ["纲手", "Gshou05"],
    "velevt8800": ["velevt8800"],
    "Q妹": ["Q妹"],
    "guestaliz": ["guestaliz"],
    "慾猫": ["慾猫", "yumaohenguaiya"],
    "木东": ["木东"],
    "羊咩咩": ["羊咩咩", "dywb"],
    "我的枪好长": ["我的枪好长", "ZzI999"],
    "xiangcaiking1": ["xiangcaiking1", "香菜", "香菜老师"],
    "桔梗": ["桔梗", "eva991009"],
    "shixiaotaone": ["shixiaotaone", "是小桃呢"],
    "米胡桃": ["米胡桃", "andmlove"],
    "linjianvhai": ["linjianvhai"],
    "daidai-77": ["daidai-77"],
    "lucy_1811": ["lucy_1811", "灵灵", "不纯学妹"],
    "Kiki_2025": ["Kiki_2025"],
    "Ema_japanese": ["Ema_japanese"],
    "Sakura_Anne": ["Sakura_Anne"],
    "shycinderella": ["shycinderella"],
    "gabbi_i": ["gabbi_i"],
    "Lucky-is-lucky": ["Lucky-is-lucky"],
    "xiaoxi---n": ["xiaoxi---n"],
    "Sunshineeve23": ["Sunshineeve23"],
    "wojuu": ["wojuu"],
    "恩智": ["恩智", "朴恩智", "PyoEunji", "Pyoapple"],
    "yuuu": ["yuuu"],
    "Pureding": ["Pureding", "퓨딩"],
    "金先生": ["金先生"],
    "LilyKoti": ["LilyKoti"],
    "孙禾颐": ["孙禾颐", "JennyPinky"],
    "Sexy yuki": ["Sexy yuki", "yy姐"],
    "namprikk": ["namprikk", "Npxvip"],
    "Telari Love": ["Telari Love", "TelariLove"],
    "Gattouz0": ["Gattouz0"],
    "Yuumeilyn": ["Yuumeilyn", "虞梅", "Meiilyn"],
    "kittyxkum": ["kittyxkum"],
    "seoldol": ["seoldol", "소立つ"],
    "bubblexgun": ["bubblexgun"],
    "Rae Lil Black": ["Rae Lil Black"],
    "Mia Malkova": ["Mia Malkova"],
    "Nicole Doshi": ["Nicole Doshi"],
    "Kendra Sunderland": ["Kendra Sunderland"],
    "bigcatmia": ["bigcatmia", "高桥尤美"],
    "juju_swing": ["juju_swing"],
    "Ada Kham": ["Ada Kham"],
    "TylerX0X": ["TylerX0X"],
    "HelloElly": ["HelloElly"],
    "aqua_ri": ["aqua_ri"],
    "TheLymia": ["TheLymia"],
    "milkimind": ["milkimind"],
    "Fantasybabe": ["Fantasybabe"],
    "BootyFrutti": ["BootyFrutti"],
    "miulio": ["miulio"],
    "ClarkandMartha": ["ClarkandMartha"],
    "comatozze": ["comatozze", "cumatozz"],
    "aerytiefling": ["aerytiefling"],
    "Bradham8": ["Bradham8"],
    "MilaPixie": ["MilaPixie"],
    "Lena Polanski": ["Lena Polanski"],
    "Loraberry": ["Loraberry"],
    "sweetiefox": ["sweetiefox"],
    "ArinaFox": ["ArinaFox"],
    "NoLube": ["NoLube"],
    "gold_grass": ["gold_grass"],
    "Diana Daniels": ["Diana Daniels"],
    "Perwoopar": ["Perwoopar"],
    "annelitt": ["annelitt"],
    "layladream": ["layladream"],
    "angela doll": ["angela doll"],
    "NatalieFlowers": ["NatalieFlowers"],
    "Tony Profane": ["Tony Profane"],
    "ThePovGod": ["ThePovGod"],
    "BIGJ": ["BIGJ", "travelvids", "TRAVELVIDS"],
    "Kendra Lust": ["Kendra Lust"],
    "Sharon White": ["Sharon White"],
    "Briana Banderas": ["Briana Banderas"],
    "Abella Danger": ["Abella Danger"],
    "Elana Bunnz": ["Elana Bunnz"],
    "Casca Akashova": ["Casca Akashova"],
    "Cristy Ren": ["Cristy Ren"],
    "LinaMigurtt": ["LinaMigurtt"],
    "AYW-21": ["AYW-21"],
    "miaomiao_kitty": ["miaomiao_kitty", "Kitten_MiMi"],
    "Kovalski": ["Kovalski"],
    "knfnc": ["knfnc"],
    "叫汤米": ["叫汤米"],
    "Sy体育生": ["Sy", "富家公子哥体育生"],
    "肌肉男阿丹": ["肌肉男阿丹", "imdanbee"],
    "西柚子": ["西柚子", "天使西柚", "ts39300"],
    "瘦猴探花": ["瘦猴探花", "瘦子探花", "瘦子探花梦幻馆"],
    "橘子w": ["橘子w"],
    "是皂皂呀": ["是皂皂呀"],
    "司宁青": ["司宁青"],
    "MissKsiaBB": ["MissKsiaBB"],
    "绘子sama": ["绘子sama"],
    "宝子子": ["宝子子"],
    "Runa": ["Runa", "ふわん"],
    "Maria Ozawa": ["Maria Ozawa"],
    "石川澪": ["石川澪", "MIDA-574"],
    "枫花恋": ["枫花恋", "田中柠檬", "楓カレン"],
    "新有菜": ["新有菜", "桥本有菜", "新ありな"],
    "天海翼": ["天海翼"],
    "新村晶": ["新村晶", "高山真由", "新村あかり"],
    "古川結愛": ["古川結愛"],
    "八掛まるちゃん": ["八掛まるちゃん"],
    "ななみ": ["ななみ", "岬ななみ", "七武ななみ"],
    "NATSUMI": ["NATSUMI", "夏美"],
    "細工師Ｘ工匠": ["細工師Ｘ工匠"],
    "僕のハメ撮り日誌": ["僕のハメ撮り日誌"],
    "の車フェラ究极": ["の車フェラ究极"],
    "Hasumi Chise": ["Hasumi Chise"],
    "香织": ["香织", "前田香織", "麻田香織", "山本香織", "綺羅羅香織"],
    "あかね": ["あかね", "新田りお", "桃果あかり", "Akari Toka"],
    "水泳部顾问": ["水泳部顾问"],
    "yn_3": ["yn_3"],
    "xxx_usg": ["xxx_usg"],
    "room_103": ["room_103"],
    "rehinf": ["rehinf"],
    "aoimaria117": ["aoimaria117"],
    "shin": ["shin", "@shin"],
    "HameMoremo": ["HameMoremo"],
    "aliceholic": ["aliceholic"],
    "eakedJP": ["eakedJP", "REIPON"],
    "Mizuki Ogata": ["Mizuki Ogata"],
    "Migoto": ["Migoto"],
    "三上悠亚": ["三上悠亚"],
    "白桃はな": ["白桃はな"],
    "麻豆": ["麻豆"],
    "寂寞老师": ["寂寞老师"],
    "一日女友体验卡": ["一日女友体验卡"],
    "花碎花": ["花碎花"],
    "脸红Dearie": ["脸红Dearie"],
    "莱伦": ["莱伦"],
    "短发加禾": ["短发加禾"],
    "房东太太": ["房东太太"],
    "Shinaryen": ["Shinaryen"],
    "冯珊珊FSS": ["冯珊珊FSS"],
    "绫香不吃香菜": ["绫香不吃香菜"],
    "玲奈lena": ["玲奈lena"],
    "Szdysq199": ["Szdysq199"],
    "XenaDreamx": ["XenaDreamx"],
    "Bibo": ["Bibo"],
    "拨乱反正夫妻": ["拨乱反正夫妻"],
    "Minami Airi": ["Minami Airi"],
    "糖心vLog": ["糖心vLog"],
    "柚萌": ["柚萌"],
    "Couplelove": ["Couplelove"],
    "萝莉控狂喜": ["萝莉控狂喜"],
    "上官太太": ["上官太太"],
    "郑欣雯": ["郑欣雯"],
    "LUMA": ["LUMA"],
    "晨汐": ["晨汐", "汐梦瑶", "lo330604"],
    "小橘娘": ["小橘娘"],
    "23kennys": ["23kennys", "kennys", "kennys548534"],
    "Miuzxc": ["Miuzxc", "miuzxc"],
    "你的娇妹妹": ["你的娇妹妹"],
    "深绿岸腐猫儿": ["深绿岸腐猫儿"],
    "峰不二子": ["峰不二子"],
    "紫蛋": ["紫蛋"],
    "91Mrber泰迪": ["91Mrber泰迪"],
    "捅主任": ["捅主任"],
    "性感的猫": ["性感的猫"],
    "北门玩家": ["北门玩家"],
    "上官夫人": ["上官夫人"],
    "银丝雀": ["银丝雀"],
    "vickybb": ["vickybb"],
    "枪哥好兄弟": ["枪哥好兄弟"],
    "原创大": ["原创大"],
    "Ellie奶咪饼干姐姐": ["Ellie奶咪饼干姐姐"],
    "imladylinn小林女士": ["imladylinn小林女士"],
    "luyuan258鹿苑": ["luyuan258鹿苑"],
    "柠檬不甜": ["柠檬不甜"],
    "小杏仁": ["小杏仁"],
    "KAER没有肌肉": ["KAER没有肌肉", "ZuiAiKaEr", "卡尔"],
    "downer8gz": ["downer8gz"],
    "Tinaislove": ["Tinaislove"],
    "pupuv1": ["pupuv1", "pupuwaifu", "gfpupu1"],
    "盛鸽小鹿": ["盛鸽小鹿"],
    "小恩雅": ["小恩雅", "米娜"],
    "黎黎小迷妹": ["黎黎小迷妹"],
    "何成琳": ["何成琳"],
    "柠檬然": ["柠檬然"],
    "妙妙子": ["妙妙子"],
    "羊腿迦妮": ["羊腿迦妮"],
    "小飞粥": ["小飞粥"],
    "苏苏Suda吖": ["苏苏Suda吖"],
    "余珠珠": ["余珠珠"],
    "是莉莉呀": ["是莉莉呀"],
    "Ada": ["Ada"],
    "小蛇": ["小蛇"],
    "小艺霖": ["小艺霖", "软软", "腿腿"],
    "jvv": ["jvv"],
    "小江": ["小江", "纯种小江", "阿江"],
    "DK": ["DK"],
    "瑜伽Even": ["瑜伽Even"],
    "不认薯小兔": ["不认薯小兔"],
    "卡莎": ["卡莎"],
    "小狗不吐骨头": ["小狗不吐骨头"],
    "杨昭": ["杨昭"],
    "林露露": ["林露露"],
    "hRQ百态": ["hRQ百态"],
}


def build_alias_index():
    """构建别名反向索引: normalized_alias -> canonical_name"""
    alias_index = {}
    for canonical, aliases in BLOGGER_ALIASES.items():
        for alias in aliases:
            norm = normalize_text(alias)
            if norm not in alias_index:
                alias_index[norm] = canonical
    return alias_index


def normalize_text(text):
    if not text:
        return text
    text = unicodedata.normalize('NFC', text)
    text = text.lower()
    text = re.sub(r'[\s_\-【】（）\[\]()·/]+', '', text)
    return text


def load_knowledge_base():
    """解析博主知识库Markdown文件，提取结构化信息"""
    kb = {}
    if not os.path.exists(KB_PATH):
        return kb
    with open(KB_PATH, 'r', encoding='utf-8') as f:
        content = f.read()
    for line in content.split('\n'):
        line = line.strip()
        if line.startswith('|') and not line.startswith('|---') and not line.startswith('| 名称'):
            parts = [p.strip() for p in line.split('|')]
            if len(parts) >= 5:
                name = parts[1].strip()
                platform = parts[2].strip()
                tags = parts[3].strip()
                notes = parts[4].strip() if len(parts) > 4 else ""
                if name and name != '名称':
                    kb[normalize_text(name)] = {
                        'name': name,
                        'platform': platform,
                        'tags': tags,
                        'notes': notes,
                    }
    return kb


def match_with_confidence(text, alias_index, kb):
    """带置信度的博主名匹配"""
    text_norm = normalize_text(text)
    results = []

    # 1. 精确别名匹配 (置信度: 1.0)
    if text_norm in alias_index:
        results.append((alias_index[text_norm], 1.0, 'exact_alias'))

    # 2. 知识库精确匹配 (置信度: 0.95)
    if text_norm in kb:
        results.append((kb[text_norm]['name'], 0.95, 'kb_exact'))

    # 3. 包含匹配 (置信度: 0.7-0.9)
    for norm_alias, canonical in alias_index.items():
        if len(norm_alias) >= 3:
            if norm_alias in text_norm:
                ratio = len(norm_alias) / len(text_norm)
                conf = 0.7 + 0.2 * ratio
                results.append((canonical, min(conf, 0.9), 'alias_contains'))
            elif text_norm in norm_alias:
                ratio = len(text_norm) / len(norm_alias)
                conf = 0.7 + 0.2 * ratio
                results.append((canonical, min(conf, 0.9), 'alias_contained'))

    # 4. 平台+标签语义匹配 (置信度: 0.5-0.7)
    for norm_name, info in kb.items():
        if norm_name in text_norm or text_norm in norm_name:
            conf = 0.5
            platform_str = info.get('platform', '').lower()
            for platform, keywords in PLATFORM_MAP.items():
                for kw in keywords:
                    if kw.lower() in text_norm and kw.lower() in platform_str.lower():
                        conf += 0.1
                        break
            results.append((info['name'], min(conf, 0.7), 'kb_semantic'))

    # 去重，保留最高置信度
    best = {}
    for canonical, conf, method in results:
        if canonical not in best or best[canonical][1] < conf:
            best[canonical] = (canonical, conf, method)

    return sorted(best.values(), key=lambda x: x[1], reverse=True)


def extract_blogger_with_kb(filename, alias_index, kb):
    """结合知识库的博主名提取"""
    from enhanced_video_matcher import extract_blogger_from_file, clean_blogger_name, is_valid_blogger_name

    raw = extract_blogger_from_file(filename)
    if not raw:
        return None, 0.0, 'none'

    raw_clean = clean_blogger_name(raw)
    matches = match_with_confidence(raw_clean, alias_index, kb)

    if matches:
        best = matches[0]
        return best[0], best[1], best[2]

    if is_valid_blogger_name(raw_clean):
        return raw_clean, 0.3, 'raw_valid'

    return None, 0.0, 'none'


def analyze_with_kb(base_path=BASE_PATH):
    """结合知识库的全面分析"""
    print("=" * 80)
    print("Knowledge-Enhanced Blogger Matcher")
    print(f"Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print("=" * 80)

    print("\n[1/4] Loading knowledge base...")
    kb = load_knowledge_base()
    print(f"  Loaded {len(kb)} entries from knowledge base")

    print("\n[2/4] Building alias index...")
    alias_index = build_alias_index()
    print(f"  Built {len(alias_index)} alias mappings")

    print("\n[3/4] Scanning files...")
    import subprocess
    try:
        result = subprocess.run(
            f'find "{base_path}" -maxdepth 3 -type f '
            f'\\( -iname "*.mp4" -o -iname "*.avi" -o -iname "*.mov" '
            f'-o -iname "*.mkv" \\) 2>/dev/null',
            shell=True, capture_output=True, text=True, timeout=600
        )
        files = result.stdout.strip().split('\n') if result.stdout.strip() else []
    except:
        files = []

    matches = defaultdict(list)
    total = 0
    matched = 0
    confidence_dist = defaultdict(int)

    for file_path in files:
        if not file_path:
            continue
        filename = os.path.basename(file_path)
        if filename.startswith('._'):
            continue
        total += 1

        blogger, conf, method = extract_blogger_with_kb(filename, alias_index, kb)
        if blogger:
            matched += 1
            confidence_dist[method] += 1
            matches[blogger].append(('file', file_path, os.path.relpath(file_path, base_path), conf))

    print(f"  Scanned {total} files, matched {matched} ({matched*100//max(total,1)}%)")
    print(f"  Confidence distribution:")
    for method, count in sorted(confidence_dist.items(), key=lambda x: -x[1]):
        print(f"    {method}: {count} ({count*100//max(matched,1)}%)")

    print("\n[4/4] Results:")
    print("=" * 80)

    for blogger in sorted(matches.keys()):
        items = matches[blogger]
        avg_conf = sum(i[3] for i in items) / len(items)
        print(f"\n📁 {blogger} (avg confidence: {avg_conf:.2f}, {len(items)} files)")
        for t, path, name, conf in items[:3]:
            print(f"   [{conf:.2f}] {name}")
        if len(items) > 3:
            print(f"   ... +{len(items) - 3} more")

    print(f"\n{'=' * 80}")
    print(f"Summary: {len(matches)} bloggers, {total} files, {matched} matched")
    print("=" * 80)

    return matches, kb, alias_index


if __name__ == "__main__":
    analyze_with_kb()
