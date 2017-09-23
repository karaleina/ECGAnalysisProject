from AF.tools.wfdbtools import CODEDICT

class Annotation(object):
    def __init__(self, time_in_samples, type_code, name, auxiliary_data=None):
        self.time_in_samples = time_in_samples
        self.type_code = type_code
        self.name = name
        self.auxiliary_data = auxiliary_data

    def __str__(self, *args, **kwargs):
        text_representation = 'sample = {sample}, code = {code}, name = {name}'.format(
            sample=self.time_in_samples, code=self.type_code, name=self.name)

        if self.auxiliary_data:
            text_representation += ', data = {}'.format(self.auxiliary_data)

        return text_representation


class AnnotationReader(object):
    def read_annotations(self, file_path):
        annotations = []

        with open(file_path, 'rb') as input_file:
            while True:
                bits = input_file.read(2)
                type_code = bits[1] >> 2
                data = ((bits[1] & 0x03) << 6) | bits[0]

                if 0 < type_code <= 49:
                    annotations.append(Annotation(data, type_code, CODEDICT[type_code]))
                elif type_code == 59:
                    raw_interval = input_file.read(4)
                    interval = raw_interval[1] << 24 | raw_interval[0] << 16 | raw_interval[3] << 8 | raw_interval[2]
                    annotations.append(Annotation(0, type_code, 'SKIP', str(interval)))
                elif type_code == 60:
                    annotations.append(Annotation(0, type_code, 'NUM', str(data)))
                elif type_code == 61:
                    annotations.append(Annotation(0, type_code, 'SUB', str(data)))
                elif type_code == 62:
                    annotations.append(Annotation(0, type_code, 'CHN', str(data)))
                elif type_code == 63:
                    information_length = data
                    if information_length % 2 == 0:
                        information = input_file.read(information_length).decode('ascii')
                    else:
                        information = input_file.read(information_length + 1)[:-1].decode('ascii')
                    annotations.append(Annotation(0, type_code, 'AUX', information))
                elif type_code == 0:
                    break
                else:
                    annotations.append(Annotation(data, type_code, 'CUSTOM'))

        return annotations
