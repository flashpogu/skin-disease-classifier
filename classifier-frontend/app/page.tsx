"use client";

import { useState } from "react";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Upload, X, CheckCircle } from "lucide-react";

interface PredictionResponse {
  predicted_label: string;
  probabilities: number[][];
  heatmap: string;
}

const DISEASE_LABELS: { [key: string]: string } = {
  akiec: "Actinic Keratoses",
  bcc: "Basal Cell Carcinoma",
  bkl: "Benign Keratosis",
  df: "Dermatofibroma",
  mel: "Melanoma",
  nv: "Melanocytic Nevi (Mole)",
  vasc: "Vascular Lesions",
};

// ✅ Replace with your own Modal URLs
const INFERENCE_URL =
  "https://zeddx-rahul--skin-disease-classifier-skinclassifier-inference.modal.run";
const REPORT_URL =
  "https://zeddx-rahul--skin-disease-classifier-skinclassifier-report.modal.run";

export default function HomePage() {
  const [isLoading, setIsLoading] = useState(false);
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);
  const [error, setError] = useState<string | null>(null);

  const [viewMode, setViewMode] = useState<"original" | "heatmap">("original");
  const [imageBase64, setImageBase64] = useState<string | null>(null);

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;

    setIsLoading(true);
    setPrediction(null);
    setError(null);
    setViewMode("original");

    const previewUrl = URL.createObjectURL(file);
    setImagePreview(previewUrl);

    const reader = new FileReader();
    reader.readAsArrayBuffer(file);
    reader.onload = async () => {
      try {
        const arrayBuffer = reader.result as ArrayBuffer;
        const base64String = btoa(
          new Uint8Array(arrayBuffer).reduce(
            (data, byte) => data + String.fromCharCode(byte),
            ""
          )
        );
        setImageBase64(base64String);

        const response = await fetch(INFERENCE_URL, {
          method: "POST",
          headers: { "Content-Type": "application/json" },
          body: JSON.stringify({ image_data: base64String }),
        });

        const data: PredictionResponse = await response.json();
        setPrediction(data);
      } catch {
        setError("❌ Something went wrong!");
      } finally {
        setIsLoading(false);
      }
    };
  };

  const handleReset = () => {
    setImagePreview(null);
    setPrediction(null);
    setError(null);
    setViewMode("original");
  };

  const getSortedProbabilities = () => {
    if (!prediction) return [];
    const probs = prediction.probabilities[0];
    const keys = Object.keys(DISEASE_LABELS);

    return probs
      .map((p, i) => ({
        key: keys[i],
        name: DISEASE_LABELS[keys[i]],
        prob: p,
      }))
      .sort((a, b) => b.prob - a.prob);
  };

  const displayImage =
    viewMode === "heatmap"
      ? `data:image/jpeg;base64,${prediction?.heatmap}`
      : imagePreview;

  return (
    <main className="min-h-screen bg-gradient-to-br from-stone-50 to-stone-100 p-8">
      <div className="mx-auto max-w-7xl">
        {/* Header */}
        <div className="mb-12 text-center">
          <h1 className="text-5xl font-light mb-2 tracking-tight text-stone-900">
            Skin Disease Classifier
          </h1>
          <p className="text-stone-600 text-lg">
            Upload an image to analyze skin lesions with AI
          </p>
        </div>

        <div className="grid gap-8 lg:grid-cols-2">
          {/* Left: Image Upload */}
          <Card className="border-stone-200 shadow-lg">
            <CardHeader>
              <CardTitle className="text-stone-800 flex gap-2 items-center">
                <Upload className="h-5 w-5" /> Upload Image
              </CardTitle>
            </CardHeader>
            <CardContent>
              {!imagePreview ? (
                <label className="flex h-64 cursor-pointer flex-col items-center justify-center rounded-lg border-2 border-dashed border-stone-300 bg-stone-50 hover:border-stone-400 hover:bg-stone-100">
                  <Upload className="h-10 w-10 text-stone-400 mb-3" />
                  <p className="text-stone-600 text-sm">Click or drop image here</p>
                  <input
                    type="file"
                    accept=".jpg,.jpeg,.png"
                    className="hidden"
                    onChange={handleFileChange}
                  />
                </label>
              ) : (
                <div className="relative">
                  <img
                    src={displayImage ?? ""}
                    className="rounded-lg border w-full object-contain"
                    style={{ maxHeight: "400px" }}
                  />
                  <Button
                    variant="destructive"
                    size="sm"
                    className="absolute top-3 right-3"
                    onClick={handleReset}
                  >
                    <X size={14} />
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>

          {/* Right: Results */}
          <Card className="border-stone-200 shadow-lg">
            <CardHeader>
              <CardTitle className="text-stone-800 flex gap-2 items-center">
                <CheckCircle className="h-5 w-5" /> Results
              </CardTitle>
            </CardHeader>
            <CardContent>
              {!prediction && !isLoading && (
                <p className="text-stone-500 text-center h-64 flex items-center justify-center">
                  Upload an image to see analysis
                </p>
              )}

              {isLoading && (
                <div className="flex justify-center py-20">
                  <div className="animate-spin rounded-full h-12 w-12 border-4 border-stone-300 border-t-stone-900"></div>
                </div>
              )}

              {error && (
                <p className="text-center text-red-600 font-medium">{error}</p>
              )}

              {prediction && (
                <div className="space-y-6">
                  {/* ✅ Toggle */}
                  <div className="flex justify-center gap-2">
                    <Button
                      size="sm"
                      className={viewMode === "original"
                        ? "bg-stone-800 text-white"
                        : "bg-stone-200 text-stone-700"}
                      onClick={() => setViewMode("original")}
                    >
                      Original
                    </Button>
                    <Button
                      size="sm"
                      className={viewMode === "heatmap"
                        ? "bg-stone-800 text-white"
                        : "bg-stone-200 text-stone-700"}
                      onClick={() => setViewMode("heatmap")}
                    >
                      Heatmap
                    </Button>
                  </div>

                  {/* ✅ Prediction */}
                  <div className="p-4 bg-white border rounded-lg">
                    <p className="text-[12px] text-stone-500">Prediction</p>
                    <h3 className="font-bold text-xl">
                      {DISEASE_LABELS[prediction.predicted_label]}
                    </h3>
                    <p className="text-xs text-stone-400">
                      ({prediction.predicted_label.toUpperCase()})
                    </p>
                  </div>

                  {/* ✅ Probabilities */}
                  <div className="space-y-2">
                    {getSortedProbabilities().map((item, idx) => (
                      <div
                        key={idx}
                        className="bg-white border rounded-md p-2"
                      >
                        <div className="flex justify-between">
                          <span className="text-sm">{item.name}</span>
                          <span className="text-sm font-medium">
                            {(item.prob * 100).toFixed(1)}%
                          </span>
                        </div>
                        <div className="w-full h-2 bg-stone-200 rounded-full mt-1">
                          <div
                            className={`h-full rounded-full ${
                              idx === 0 ? "bg-green-700" : "bg-stone-400"
                            }`}
                            style={{ width: `${item.prob * 100}%` }}
                          ></div>
                        </div>
                      </div>
                    ))}
                  </div>

                  {/* ✅ PDF Button */}
                  <Button
                    className="w-full bg-blue-700 text-white"
                    onClick={async () => {
                      if (!imageBase64) return;

                      const res = await fetch(REPORT_URL, {
                        method: "POST",
                        headers: { "Content-Type": "application/json" },
                        body: JSON.stringify({
                          image_data: imageBase64,
                        }),
                      });

                      const { pdf_report } = await res.json();
                      const link = document.createElement("a");
                      link.href = `data:application/pdf;base64,${pdf_report}`;
                      link.download = "AI_Skin_Report.pdf";
                      link.click();
                    }}
                  >
                    📄 Download Report
                  </Button>

                  {/* Reset */}
                  <Button
                    variant="outline"
                    className="w-full"
                    onClick={handleReset}
                  >
                    Analyze New Image
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        <p className="text-[11px] text-center text-stone-500 mt-6">
          ⚠ Not a medical diagnosis. Consult a dermatologist.
        </p>
      </div>
    </main>
  );
}
