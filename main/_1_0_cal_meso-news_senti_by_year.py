from transformers import AutoModelForCausalLM, AutoTokenizer
import os
import sys
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from tqdm import tqdm
import logging
import json
import torch

def setup_logging(log_file_path):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    file_handler = logging.FileHandler(log_file_path, mode='a')
    file_handler.setLevel(logging.INFO)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')
    file_handler.setFormatter(formatter)
    console_handler.setFormatter(formatter)

    logger.addHandler(file_handler)
    logger.addHandler(console_handler)

def main(data_path, model_id):
    json_path = data_path + '.json'
    data = json.load(open(json_path, 'r', encoding='utf-8'))
    model = AutoModelForCausalLM.from_pretrained(
    model_id,
    torch_dtype="auto",
    device_map="cuda:7"
    )
    tokenizer = AutoTokenizer.from_pretrained(model_id)
    iter = 0
    for item in tqdm(data, desc=f'{json_path}'):
        body = item['body']
        messages = [
        {"role": "system", "content":"""
        你是一个专业的债券市场分析专家。接下来，你将获得一系列有关债券市场的新闻数据。你的任务是基于新闻内容，判断其对应的情绪数值。情绪数值只有-1，0，1三种离散的类别标签，其中：
        a) -1表示悲观；
        b) 0表示中性；
        c) 1表示乐观。
        具体标准如下：
        1. 悲观：情绪数值为-1, 表示市场预期可能恶化或存在较高的风险。具体表现为：
        1) 经济基本面：新闻提到经济增长放缓、企业盈利下降、失业率上升等负面经济指标。
        2) 利率与货币政策：预期加息幅度增加或更长时间维持高利率，导致债券价格下跌。
        3) 市场流动性：提到市场流动性紧张或投资者避险情绪升温。
        4) 风险事件：涉及债务违约、信用评级下调、重大地缘政治冲突等信息。
        5) 范围：与宏观的金融市场直接或间接相关，尤其与债券市场强相关，直接或间接地反映了市场情绪
        6) 关键词：下跌、恶化、风险升高、压力、违约、下调。
        2. 中性：情绪数值为0，新闻对市场的影响较为平衡，既没有明确的负面影响，也没有明显的积极推动因素。具体表现为：
        1) 经济基本面：新闻描述经济数据符合预期，变化幅度有限。
        2) 利率与货币政策：提到货币政策保持不变或市场已充分消化相关政策预期。
        3) 市场流动性：市场情绪稳定，买卖双方力量均衡，波动性较低。
        4) 风险事件：无重大风险事件或影响较小。
        5) 范围：所提及的内容基本与金融市场无关，不反映宏观的金融市场尤其是债券市场环境
        6) 关键词：平稳、维持、符合预期、观望、不变。
        3. 乐观：情绪数值为1，反映市场预期改善或有积极的推动因素。具体表现为：
        1) 经济基本面：新闻提到经济增长提速、企业盈利改善、就业数据向好等正面经济指标。
        2) 利率与货币政策：预期降息或宽松货币政策出台，可能推升债券价格。
        3) 市场流动性：提到市场流动性充裕或投资者风险偏好增加。
        4) 风险事件：涉及债务问题缓解、信用评级上调、地缘政治风险缓和等信息。
        5) 范围：与宏观的金融市场直接或间接相关，尤其与债券市场强相关，直接或间接地反映了市场情绪
        6) 关键词：上升、改善、缓解、推动、宽松、上调。
        输出要求：
        • a) 首先识别新闻的主题，确定其是否与债券市场、经济基本面、利率政策、市场流动性等相关
        • b) 然后判断新闻是否包含积极、消极或中性的市场影响因素
        • c) 结合关键词（如‘改善’、‘上升’=1；‘恶化’、‘违约’=-1；‘稳定’、‘符合预期’=0）进行最终判断
        • d) 最后需要返回单一的情绪数值，其取值空间为{-1,0,1}。
        • e) 不允许输出任何额外内容或解释。下面是一些示例的输入输出例子：
        例子1：
        输入：证监会发布《上市公司监管指引第10号——市值管理》。《指引》要求上市公司以提高公司质量为基础，提升经营效率和盈利能力，并结合实际情况依法合规运用并购重组、股权激励、员工持股计划、现金分红、投资者关系管理、信息披露、股份回购等方式，推动上市公司投资价值合理反映上市公司质量。《指引》指出，上市公司应当牢固树立回报股东意识，必要时积极采取措施提振投资者信心。
        思考过程：证监会属于政府机关，它出台措施鼓励企业提升经营效率和盈利能力，推动上市公司投资价值合理反映上市公司质量，这意味着政府部门对上市公司的监管力度加大，对上市公司的要求提高，市场预期可能会有所改善，情绪数值为1。
        输出：1
        例子2：
        输入：财政部、税务总局发布调整出口退税政策的公告，12月1日起实施。根据公告，取消铝材、铜材以及化学改性的动、植物或微生物油、脂等产品出口退税。将部分成品油、光伏、电池、部分非金属矿物制品的出口退税率由13%下调至9%。
        思考过程：政府部分出台了调整出口退税政策的公告，取消了部分产品的出口退税，同时将部分产品的出口退税率下调。这意味着相关产品的出口退税优惠程度降低，可能会对相关行业的出口带来一定的影响，市场预期可能会有所恶化，因此总体上是一个悲观面消息，情绪数值为-1。
        输出：-1
        例子3：
        输入：证监会近期对9家经营机构落实《证券基金经营机构董事、监事、高级管理人员及从业人员监督管理办法》情况开展了现场检查，并依法对相关机构违反规定的行为采取了监管措施。从检查情况看，仍有部分机构对办法理解不到位，执行有偏差。
        思考过程：证监会对政策的突击检查，发现部分机构对办法理解不到位，执行有偏差，这意味着部分机构存在违规行为，市场预期可能会有所恶化，情绪数值为-1。
        输出：-1
        例子4：
        输入：财政部国债招标室11月15日公布2024年超长期特别国债（六期）第三次续发行招标情况，标志着今年1万亿元超长期特别国债发行完毕。这意味着11月中旬至12月底，地方政府需要基本完成2万亿元隐债置换相关债券发行工作。
        思考过程：经济部门按时间线正常的工作流程，没有体现强烈的情绪倾向，市场预期可能会保持稳定，情绪数值为0。
        输出：0
        例子5：
        输入：有消息称，上海市公安局浦东分局交警支队近日颁发了首批无人驾驶装备识别标牌，新车牌设计为淡蓝色与白色结合，牌照开头为地区简称，后面是字母与数字组合，上方清晰标注“无人装备”。不少网友将无人驾驶装备识别标牌与无人驾驶汽车关联起来，也有自媒体称这是“无人驾驶车专属车牌”。11月15日，浦东新区相关部门及相关企业方面表示，上述理解为误读，无人驾驶装备并不是网友以为的载人汽车，而是指无人配送的低速轮式装备。
        思考过程：政府部门颁发了无人驾驶装备识别标牌，但是相关部门澄清了网友的误读，这意味着政府部门对无人驾驶装备的管理有所加强，但并不是网友所猜测的无人驾驶汽车。整体上，这是一个中性消息，情绪数值为0。
         输出：0
        例子6：
        输入：为积极响应国家关于壮大“耐心资本”的号召，在上海市委、市政府关心和指导下，生物医药先导母基金今年正式成立。母基金将加快布局产业链短板和关键核心领域，加大投早投小力度，着力提升产业整体能级。
        思考过程：当地政府部门积极采取措施鼓励生物医药产业发展，对生物医药产业的支持力度加大，市场预期可能会有所改善，情绪数值为1。
        输出：1
        例子7：
        输入：2024中国医药工业发展大会与上海国际生物医药产业周11月16日在上海举行。大会公布数据显示，“十四五”以来，我国国产创新药数量和质量齐升，共有113个国产创新药获批上市，是“十三五”获批新药数量的2.8倍，市场规模达1000亿元。
        思考过程：统计数据显示近期医药产业产值和规模翻倍，产业向好，市场预期可能会有所改善，情绪数值为1。
        输出：1
        例子8：
        输入：上海中心气象台2024年11月17日08时20分发布大风蓝色预警信号：受较强的冷空气影响，预计未来24小时内本市将出现陆地最大阵风7-8级，沿江沿海地区8-9级的偏北大风。
        思考过程：地区出现极端天气，可能影响相关产业的生产秩序和人们的生活节奏，市场预期可能会有所恶化，情绪数值为-1。
        输出：-1
        例子9：
        输入：财联社5月20日电，今日在2023清华五道口全球金融论坛上，国家外汇管理局外汇研究中心主任丁志杰表示，发达国家再次面临着物价稳定和金融稳定的两难选择。这次通胀的成因是复杂的，有新冠肺炎的因素，有需求过热的因素，有全球产业链供应链因素，也有地缘冲突的因素，这些因素叠加导致全球大宗商品价格快速上涨，目前虽然已有回落，但欧美的通胀黏性仍然较为明显。欧美的应对政策是选择了类似八十年代的加息做法，欧美银行均在5月份持续加息，尤其是美联储已经自2022以来连续十次加息，共计500个基点。加息政策的副作用越来越明显，有可能引发欧美银行业危机。
        思考过程：政府部门承认近期由于疫情等多因素的作用出现了通货膨胀，根据欧美国家的经验，加息可能会导致银行业危机，市场预期可能会有所恶化，情绪数值为-1。
        输出：-1
        例子10：
        输入：佐力药业接受机构调研时表示，公司乌灵胶囊此前进入了广东、陕西、河北、湖北4省新冠肺炎中医药防治用药目录，公司销售团队有推出“阳康套餐”，乌灵胶囊主打治疗失眠、焦虑，百令片具有补肺肾、益精气作用，用于肺肾两虚引起的咳嗽等的治疗。
        思考过程：某公司的产品进入了新市场，并公布了产品疗效，但对相关行业没有明显影响，市场预期可能会保持稳定，情绪数值为0。
        输出：0
         """},
        {"role": "user", "content": body}
        ]

        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True
        )

        model_inputs = tokenizer([text], return_tensors="pt").to(model.device)

        generated_ids = model.generate(
            **model_inputs,
            max_new_tokens=2
        )
        generated_ids = [
            output_ids[len(input_ids):] for input_ids, output_ids in zip(model_inputs.input_ids, generated_ids)
        ]
        response = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]

        item['sentiment'] = response
        iter += 1
        if iter % 5 == 0:
            torch.cuda.empty_cache()
    json.dump(data, open(f'{data_path}_senti.json', 'w', encoding='utf-8'), ensure_ascii=False, indent=4)


if __name__ == '__main__':
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    model_id_list = [
        # "Qwen/Qwen2.5-1.5B-Instruct",
        "Qwen/Qwen2.5-3B-Instruct",
        # "Qwen/Qwen2.5-7B-Instruct",
        # "Qwen/Qwen2-7B-Instruct"
    ]
    data_path_list = ["../raw_data/data_2013",
                      "../raw_data/data_2014",
                      "../raw_data/data_2015",
                      "../raw_data/data_2016",
                      "../raw_data/data_2017",
                      "../raw_data/data_2018",
                      "../raw_data/data_2019",
                      "../raw_data/data_2021",
                      "../raw_data/data_2022",
                      "../raw_data/data_2023",
        "/data2/lyw/text/data/data_2020",
    ]

    for data_path in data_path_list:
        for i in range(len(model_id_list)):
            model_id = model_id_list[i]
            main(data_path, model_id)

    filepath_list = [
        '../raw_data/data_2013_senti.json',
        '../raw_data/data_2014_senti.json',
        '../raw_data/data_2015_senti.json',
        '../raw_data/data_2016_senti.json',
        '../raw_data/data_2017_senti.json',
        '../raw_data/data_2018_senti.json',
        '../raw_data/data_2019_senti.json',
        '../raw_data/data_2021_senti.json',
        '../raw_data/data_2022_senti.json',
        '../raw_data/data_2023_senti.json',
    ]

    # Initialize a list to hold the merged data
    merged_data = []

    # Loop through each file and load its content
    for filepath in tqdm(filepath_list):
        try:
            with open(filepath, 'r') as file:
                data = json.load(file)  # Load JSON data
                if isinstance(data, list):
                    merged_data.extend(data)  # Extend the list if the JSON is a list
                else:
                    merged_data.append(data)  # Append the data if it is not a list
        except Exception as e:
            print(f"Error reading {filepath}: {e}")

    # Write the merged data to a new JSON file
    output_filepath = '../raw_data/merged_data_except_2020_senti.json'
    with open(output_filepath, 'w', encoding="utf-8") as output_file:
        json.dump(merged_data, output_file, indent=4)

    print(f"Merged JSON data has been saved to {output_filepath}")


    '''
            你是一个专业的债券市场分析专家。接下来，你将获得一系列有关债券市场的新闻数据。你的任务是基于新闻内容，判断其对应的情绪数值。情绪数值只有-1，0，1三种离散的类别标签，其中：
        a) -1表示悲观；
        b) 0表示中性；
        c) 1表示乐观。
        具体标准如下：
        1. 悲观：情绪数值为-1, 表示市场预期可能恶化或存在较高的风险。具体表现为：
        1) 经济基本面：新闻提到经济增长放缓、企业盈利下降、失业率上升等负面经济指标。
        2) 利率与货币政策：预期加息幅度增加或更长时间维持高利率，导致债券价格下跌。
        3) 市场流动性：提到市场流动性紧张或投资者避险情绪升温。
        4) 风险事件：涉及债务违约、信用评级下调、重大地缘政治冲突等信息。
        5) 范围：与宏观的金融市场直接或间接相关，尤其与债券市场强相关，直接或间接地反映了市场情绪
        6) 关键词：下跌、恶化、风险升高、压力、违约、下调。
        2. 中性：情绪数值为0，新闻对市场的影响较为平衡，既没有明确的负面影响，也没有明显的积极推动因素。具体表现为：
        1) 经济基本面：新闻描述经济数据符合预期，变化幅度有限。
        2) 利率与货币政策：提到货币政策保持不变或市场已充分消化相关政策预期。
        3) 市场流动性：市场情绪稳定，买卖双方力量均衡，波动性较低。
        4) 风险事件：无重大风险事件或影响较小。
        5) 范围：所提及的内容基本与金融市场无关，不反映宏观的金融市场尤其是债券市场环境
        6) 关键词：平稳、维持、符合预期、观望、不变。
        3. 乐观：情绪数值为1，反映市场预期改善或有积极的推动因素。具体表现为：
        1) 经济基本面：新闻提到经济增长提速、企业盈利改善、就业数据向好等正面经济指标。
        2) 利率与货币政策：预期降息或宽松货币政策出台，可能推升债券价格。
        3) 市场流动性：提到市场流动性充裕或投资者风险偏好增加。
        4) 风险事件：涉及债务问题缓解、信用评级上调、地缘政治风险缓和等信息。
        5) 范围：与宏观的金融市场直接或间接相关，尤其与债券市场强相关，直接或间接地反映了市场情绪
        6) 关键词：上升、改善、缓解、推动、宽松、上调。
        输出要求：
        • a) 首先识别新闻的主题，确定其是否与债券市场、经济基本面、利率政策、市场流动性等相关
        • b) 然后判断新闻是否包含积极、消极或中性的市场影响因素
        • c) 结合关键词（如‘改善’、‘上升’=1；‘恶化’、‘违约’=-1；‘稳定’、‘符合预期’=0）进行最终判断
        • d) 最后需要返回单一的情绪数值，其取值空间为{-1,0,1}。
        • e) 不允许输出任何额外内容或解释。下面是一些示例的输入输出例子：
        例子1：
        输入：证监会发布《上市公司监管指引第10号——市值管理》。《指引》要求上市公司以提高公司质量为基础，提升经营效率和盈利能力，并结合实际情况依法合规运用并购重组、股权激励、员工持股计划、现金分红、投资者关系管理、信息披露、股份回购等方式，推动上市公司投资价值合理反映上市公司质量。《指引》指出，上市公司应当牢固树立回报股东意识，必要时积极采取措施提振投资者信心。
        输出：1
        例子2：
        输入：财政部、税务总局发布调整出口退税政策的公告，12月1日起实施。根据公告，取消铝材、铜材以及化学改性的动、植物或微生物油、脂等产品出口退税。将部分成品油、光伏、电池、部分非金属矿物制品的出口退税率由13%下调至9%。
        输出：-1
        例子3：
        输入：证监会近期对9家经营机构落实《证券基金经营机构董事、监事、高级管理人员及从业人员监督管理办法》情况开展了现场检查，并依法对相关机构违反规定的行为采取了监管措施。从检查情况看，仍有部分机构对办法理解不到位，执行有偏差。
        输出：-1
        例子4：
        输入：财政部国债招标室11月15日公布2024年超长期特别国债（六期）第三次续发行招标情况，标志着今年1万亿元超长期特别国债发行完毕。这意味着11月中旬至12月底，地方政府需要基本完成2万亿元隐债置换相关债券发行工作。
        输出：0
        例子5：
        输入：有消息称，上海市公安局浦东分局交警支队近日颁发了首批无人驾驶装备识别标牌，新车牌设计为淡蓝色与白色结合，牌照开头为地区简称，后面是字母与数字组合，上方清晰标注“无人装备”。不少网友将无人驾驶装备识别标牌与无人驾驶汽车关联起来，也有自媒体称这是“无人驾驶车专属车牌”。11月15日，浦东新区相关部门及相关企业方面表示，上述理解为误读，无人驾驶装备并不是网友以为的载人汽车，而是指无人配送的低速轮式装备。
        输出：0
        例子6：
        输入：为积极响应国家关于壮大“耐心资本”的号召，在上海市委、市政府关心和指导下，生物医药先导母基金今年正式成立。母基金将加快布局产业链短板和关键核心领域，加大投早投小力度，着力提升产业整体能级。
        输出：1
        例子7：
        输入：2024中国医药工业发展大会与上海国际生物医药产业周11月16日在上海举行。大会公布数据显示，“十四五”以来，我国国产创新药数量和质量齐升，共有113个国产创新药获批上市，是“十三五”获批新药数量的2.8倍，市场规模达1000亿元。
        输出：1
        例子8：
        输入：上海中心气象台2024年11月17日08时20分发布大风蓝色预警信号：受较强的冷空气影响，预计未来24小时内本市将出现陆地最大阵风7-8级，沿江沿海地区8-9级的偏北大风。
        输出：-1
        例子9：
        输入：财联社5月20日电，今日在2023清华五道口全球金融论坛上，国家外汇管理局外汇研究中心主任丁志杰表示，发达国家再次面临着物价稳定和金融稳定的两难选择。这次通胀的成因是复杂的，有新冠肺炎的因素，有需求过热的因素，有全球产业链供应链因素，也有地缘冲突的因素，这些因素叠加导致全球大宗商品价格快速上涨，目前虽然已有回落，但欧美的通胀黏性仍然较为明显。欧美的应对政策是选择了类似八十年代的加息做法，欧美银行均在5月份持续加息，尤其是美联储已经自2022以来连续十次加息，共计500个基点。加息政策的副作用越来越明显，有可能引发欧美银行业危机。
        输出：-1
        例子10：
        输入：佐力药业接受机构调研时表示，公司乌灵胶囊此前进入了广东、陕西、河北、湖北4省新冠肺炎中医药防治用药目录，公司销售团队有推出“阳康套餐”，乌灵胶囊主打治疗失眠、焦虑，百令片具有补肺肾、益精气作用，用于肺肾两虚引起的咳嗽等的治疗。
        输出：0
    '''


    
    

