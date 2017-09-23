## Przykład wypisania wszystkich anotacji


from anotacje.annotation_reader import AnnotationReader



annotations_reader = AnnotationReader()
annotations = annotations_reader.read_annotations('downloads/04015.atr')

for annotation in annotations:
    print(annotation)