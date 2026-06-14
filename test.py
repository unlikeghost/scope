
from scope.scope_poly import SCoPEPolygon
from scope.utils import plot_polygon_prediction

supports = {0: ["Holi", "Hola", "alo"], 1: ["Adios", "Adio", "adiou"]}

query = "holi"

model = SCoPEPolygon(
    compressors=["gzip", "bz2"],
    keep_similar=True,
    dissimilarity_metrics=['ncd', 'cdm']
)

pred = model.predict(kw_samples=[supports], queries=[query])

print(pred[0].scores, pred[0].predicted_class)

plot_polygon_prediction(prediction=pred[0], save_path='test1.png')