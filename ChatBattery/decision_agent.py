from .domain_agent import Domain_Agent


class Decision_Agent:
    @staticmethod
    def decide_one_pair(input_formula, output_formula):
        # 此方法现在基于 domain_agent 中的更改，默认处理钠离子电池
        input_value = Domain_Agent.calculate_theoretical_capacity(input_formula)
        output_value = Domain_Agent.calculate_theoretical_capacity(output_formula)
        return input_value, output_value, output_value > input_value * 1

    @staticmethod
    def decide_pairs(input_formula, output_formula_list):
        answer_list = []
        for output_formula in output_formula_list:
            input_value, output_value, answer = Decision_Agent.decide_one_pair(input_formula, output_formula)
            answer_list.append([output_formula, output_value, answer])
        return answer_list
