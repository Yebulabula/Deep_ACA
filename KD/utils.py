def teacher_selector(teachers, idx):
    return teachers[idx]

def output_selector(outputs, answers, idx):
    return [outputs[i] for i in idx], [answers[i] for i in idx]
