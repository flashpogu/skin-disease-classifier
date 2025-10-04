"use client";

import { useState } from "react";
import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import { Card, CardContent, CardHeader, CardTitle } from "@/components/ui/card";
import { Upload, X, AlertCircle, CheckCircle } from "lucide-react";

interface PredictionResponse {
  predicted_label: string;
  probabilities: number[][];
}

// Common skin disease labels mapping
const DISEASE_LABELS: { [key: string]: string } = {
  "nv": "Melanocytic Nevi (Mole)",
  "mel": "Melanoma",
  "bkl": "Benign Keratosis",
  "bcc": "Basal Cell Carcinoma",
  "akiec": "Actinic Keratoses",
  "vasc": "Vascular Lesions",
  "df": "Dermatofibroma",
};

export default function HomePage() {
  const [isLoading, setIsLoading] = useState(false);
  const [fileName, setFileName] = useState("");
  const [imagePreview, setImagePreview] = useState<string | null>(null);
  const [error, setError] = useState<string | null>(null);
  const [prediction, setPrediction] = useState<PredictionResponse | null>(null);

  const handleFileChange = async (event: React.ChangeEvent<HTMLInputElement>) => {
    const file = event.target.files?.[0];
    if (!file) return;
    
    setFileName(file.name);
    setIsLoading(true);
    setError(null);
    setPrediction(null);

    // Create image preview
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

        const response = await fetch(
          "https://zeddx-rahul--skin-disease-classifier-skinclassifier-inference.modal.run",
          {
            method: "POST",
            headers: { "Content-Type": "application/json" },
            body: JSON.stringify({ image_data: base64String }),
          }
        );

        if (!response.ok) {
          throw new Error(`API error: ${response.statusText}`);
        }

        const data: PredictionResponse = await response.json();
        setPrediction(data);
        console.log(data)
      } catch (error) {
        setError(
          error instanceof Error ? error.message : "An unknown error occurred."
        );
      } finally {
        setIsLoading(false);
      }
    };
    
    reader.onerror = () => {
      setError("Failed to read the file.");
      setIsLoading(false);
    };
  };

  const handleReset = () => {
    setFileName("");
    setImagePreview(null);
    setError(null);
    setPrediction(null);
    setIsLoading(false);
  };

  const getConfidenceColor = (probability: number) => {
    if (probability >= 0.8) return "text-green-600 bg-green-50 border-green-200";
    if (probability >= 0.5) return "text-yellow-600 bg-yellow-50 border-yellow-200";
    return "text-orange-600 bg-orange-50 border-orange-200";
  };

  const getConfidenceLevel = (probability: number) => {
    if (probability >= 0.8) return "High Confidence";
    if (probability >= 0.5) return "Medium Confidence";
    return "Low Confidence";
  };

  // Get sorted probabilities with labels
  const getSortedProbabilities = () => {
    if (!prediction) return [];
    
    const labels = Object.keys(DISEASE_LABELS);
    const probs = prediction.probabilities[0];
    
    return probs
      .map((prob, idx) => ({
        label: labels[idx] || `Class ${idx + 1}`,
        fullName: DISEASE_LABELS[labels[idx]] || `Unknown Class ${idx + 1}`,
        probability: prob,
      }))
      .sort((a, b) => b.probability - a.probability);
  };

  const maxProbability = prediction 
    ? Math.max(...prediction.probabilities[0]) 
    : 0;

  return (
    <main className="min-h-screen bg-gradient-to-br from-stone-50 to-stone-100 p-8">
      <div className="mx-auto max-w-7xl">
        <div className="mb-12 text-center">
          <h1 className="mb-4 text-5xl font-light tracking-tight text-stone-900">
            Skin Disease Classifier
          </h1>
          <p className="text-lg text-stone-600">
            Upload an image to get AI-powered predictions for skin conditions
          </p>
        </div>

        <div className="grid gap-8 lg:grid-cols-2">
          {/* Upload Section */}
          <Card className="border-stone-200 shadow-lg">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-stone-800">
                <Upload className="h-5 w-5" />
                Upload Image
              </CardTitle>
            </CardHeader>
            <CardContent>
              {!imagePreview ? (
                <div className="flex flex-col items-center">
                  <div className="relative inline-block w-full">
                    <input
                      type="file"
                      accept=".jpg,.jpeg,.png"
                      id="file-upload"
                      onChange={handleFileChange}
                      disabled={isLoading}
                      className="absolute inset-0 h-full w-full cursor-pointer opacity-0"
                    />
                    <div className="flex h-64 flex-col items-center justify-center rounded-lg border-2 border-dashed border-stone-300 bg-stone-50 transition hover:border-stone-400 hover:bg-stone-100">
                      <Upload className="mb-4 h-12 w-12 text-stone-400" />
                      <p className="mb-2 text-sm font-medium text-stone-700">
                        Click to upload or drag and drop
                      </p>
                      <p className="text-xs text-stone-500">
                        JPG, JPEG, or PNG (MAX. 10MB)
                      </p>
                    </div>
                  </div>
                </div>
              ) : (
                <div className="relative">
                  <img
                    src={imagePreview}
                    alt="Uploaded preview"
                    className="h-auto w-full rounded-lg border border-stone-200 object-contain"
                    style={{ maxHeight: '400px' }}
                  />
                  <Button
                    onClick={handleReset}
                    size="sm"
                    variant="destructive"
                    className="absolute right-2 top-2"
                  >
                    <X className="h-4 w-4" />
                  </Button>
                  {fileName && (
                    <Badge
                      variant="secondary"
                      className="mt-3 bg-stone-200 text-stone-700"
                    >
                      {fileName}
                    </Badge>
                  )}
                </div>
              )}
            </CardContent>
          </Card>

          {/* Results Section */}
          <Card className="border-stone-200 shadow-lg">
            <CardHeader>
              <CardTitle className="flex items-center gap-2 text-stone-800">
                <CheckCircle className="h-5 w-5" />
                Analysis Results
              </CardTitle>
            </CardHeader>
            <CardContent>
              {!prediction && !error && !isLoading && (
                <div className="flex h-64 items-center justify-center text-center">
                  <div>
                    <AlertCircle className="mx-auto mb-3 h-12 w-12 text-stone-300" />
                    <p className="text-stone-500">
                      Upload an image to see predictions
                    </p>
                  </div>
                </div>
              )}

              {isLoading && (
                <div className="flex h-64 items-center justify-center">
                  <div className="text-center">
                    <div className="mb-4 inline-block h-12 w-12 animate-spin rounded-full border-4 border-stone-200 border-t-stone-600"></div>
                    <p className="text-stone-600">Analyzing image...</p>
                  </div>
                </div>
              )}

              {error && (
                <Card className="border-red-200 bg-red-50">
                  <CardContent className="pt-6">
                    <div className="flex items-start gap-3">
                      <AlertCircle className="h-5 w-5 text-red-600" />
                      <div>
                        <p className="font-medium text-red-800">Error</p>
                        <p className="text-sm text-red-600">{error}</p>
                      </div>
                    </div>
                  </CardContent>
                </Card>
              )}

              {prediction && (
                <div className="space-y-6">
                  {/* Main Prediction */}
                  <div
                    className={`rounded-lg border-2 p-6 ${getConfidenceColor(
                      maxProbability
                    )}`}
                  >
                    <p className="mb-2 text-sm font-medium">Predicted Condition</p>
                    <h3 className="mb-1 text-2xl font-bold">
                      {DISEASE_LABELS[prediction.predicted_label] || prediction.predicted_label}
                    </h3>
                    <p className="mb-3 text-sm opacity-75">
                      ({prediction.predicted_label.toUpperCase()})
                    </p>
                    <div className="flex items-center justify-between">
                      <span className="text-sm font-medium">
                        {getConfidenceLevel(maxProbability)}
                      </span>
                      <span className="text-2xl font-bold">
                        {(maxProbability * 100).toFixed(2)}%
                      </span>
                    </div>
                  </div>

                  {/* All Predictions */}
                  <div>
                    <p className="mb-4 text-sm font-medium text-stone-700">
                      All Predictions (Sorted by Confidence)
                    </p>
                    <div className="space-y-3">
                      {getSortedProbabilities().map((item, idx) => (
                        <div key={idx} className="rounded-lg border border-stone-200 bg-white p-4">
                          <div className="mb-2 flex items-start justify-between">
                            <div className="flex-1">
                              <p className="font-medium text-stone-800">
                                {item.fullName}
                              </p>
                              <p className="text-xs text-stone-500">
                                {item.label.toUpperCase()}
                              </p>
                            </div>
                            <Badge
                              variant={idx === 0 ? "default" : "secondary"}
                              className={idx === 0 ? "bg-stone-800" : ""}
                            >
                              {(item.probability * 100).toFixed(2)}%
                            </Badge>
                          </div>
                          <div className="h-2 w-full overflow-hidden rounded-full bg-stone-100">
                            <div
                              className={`h-full rounded-full transition-all ${
                                idx === 0 ? "bg-stone-800" : "bg-stone-400"
                              }`}
                              style={{ width: `${item.probability * 100}%` }}
                            ></div>
                          </div>
                        </div>
                      ))}
                    </div>
                  </div>

                  <Button
                    onClick={handleReset}
                    variant="outline"
                    className="w-full border-stone-300"
                  >
                    Analyze Another Image
                  </Button>
                </div>
              )}
            </CardContent>
          </Card>
        </div>

        {/* Disclaimer */}
        <Card className="mt-8 border-amber-200 bg-amber-50">
          <CardContent className="pt-6">
            <div className="flex items-start gap-3">
              <AlertCircle className="h-5 w-5 text-amber-600" />
              <div>
                <p className="font-medium text-amber-800">Medical Disclaimer</p>
                <p className="text-sm text-amber-700">
                  This tool is for educational purposes only and should not be used
                  as a substitute for professional medical advice, diagnosis, or
                  treatment. Always consult a qualified healthcare provider for any
                  skin conditions or concerns.
                </p>
              </div>
            </div>
          </CardContent>
        </Card>
      </div>
    </main>
  );
}