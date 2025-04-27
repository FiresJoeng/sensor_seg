import code_matching
import index
import model_matching


def print_connected_codes(result_dict):
    """
    将结果字典中所有的code字段连接起来并打印

    :param result_dict: 匹配结果字典
    """
    # 收集所有有效的code值
    codes = []
    for key, value in result_dict.items():
        if value is not None and 'code' in value:
            codes.append(value.get('code'))

    # 连接所有code并打印
    connected_code = ''.join(codes)
    print("\n推荐产品代码:")
    print(connected_code)


if __name__ == "__main__":
    # 示例输入参数 (与code_matching.py中相同)
    input_params = {
        '输出信号': '4~20mA DC',
        '变 送器': 'YTA系列710',
        '说明书语言': '英语',
        '传感器输入': '双支输入',
        '壳体代码': '不锈钢',
        '接线口': '1/2NPT"（F）',
        '内置指示器': '液晶数显LCD',
        '安装支架': '2"管垂直安装',
        'NEPSI': 'GB3836.1-2010、GB3836.2-2010、GB12476.1-2013、GB12476.5-2013'
    }

    # 从索引加载模型并匹配标准参数
    main_model, csv_files = index.main(
        input_params, "libs/standard/transmitter/index.json"
    )
    matched_dict = model_matching.match_standard_params(
        input_params, csv_files
    )

    # 进行模糊匹配并获取最终结果
    matcher = code_matching.CodeMatcher()
    result = matcher.match_code(input_params, matched_dict)

    # 使用csv_files列表的第一个值作为参考CSV进行排序
    refer_csv = csv_files[0] if csv_files else None
    if refer_csv:
        sorted_result = code_matching.sort_code(result, refer_csv)
    else:
        sorted_result = result

    # 打印连接后的代码
    print_connected_codes(sorted_result)
