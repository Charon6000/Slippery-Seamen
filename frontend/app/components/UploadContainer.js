'use client'

import { useState, useRef } from "react";
import Image from "next/image";

export default function UploadComponent() {
    const [selectedFiles, setSelectedFiles] = useState([]);
    const fileInputRef = useRef(null);

    const handleChange = (e) => {
        setSelectedFiles(prev => [...prev, ...Array.from(e.target.files)]);
    };

    const handleUpload = async () => {
        if (selectedFiles.length === 0) return;

        const form = new FormData();
        selectedFiles.forEach((file) => {
            form.append("files", file);
        });

        try {
            const res = await fetch("localhost:5000/", {
                method: "POST",
                body: form,
            });

            const json = await res.json();
            console.log("Server response:", json);
            alert(json.message || "Upload complete");
        } catch (err) {
            console.error("Upload error:", err);
            alert("Upload failed");
        }
    };

    return (
        <div className="flex justify-center items-center pt-44">
            <div className="flex flex-col gap-y-4">

                {selectedFiles.length === 0 && (
                    <div className="flex items-center justify-center w-full mt-5">
                        <label
                            htmlFor="dropzone-file"
                            className="flex flex-col items-center justify-center p-3 w-full h-36 border-2 border-gray-300 border-dashed rounded-lg cursor-pointer bg-[#ffffff] hover:bg-[#f9f9f9]"
                        >
                            <div className="flex flex-col items-center justify-center pt-5 pb-6">
                                <svg className="w-8 h-8 mb-4 text-[#7b7b7b] dark:text-[#9b9b9b]" aria-hidden="true" xmlns="http://www.w3.org/2000/svg" fill="none" viewBox="0 0 20 16">
                                    <path stroke="currentColor" strokeLinecap="round" strokeLinejoin="round" strokeWidth="2" d="M13 13h3a3 3 0 0 0 0-6h-.025A5.56 5.56 0 0 0 16 6.5 5.5 5.5 0 0 0 5.207 5.021C5.137 5.017 5.071 5 5 5a4 4 0 0 0 0 8h2.167M10 15V6m0 0L8 8m2-2 2 2" />
                                </svg>
                                <p className="mb-2 text-sm text-[#7b7b7b] dark:text-[#9b9b9b]"><span className="font-semibold">Click to upload</span> or drag and drop</p>
                                <p className="text-xs text-[#7b7b7b] dark:text-[#9b9b9b]">pdf, docx, doc (MAX.14MB)</p>
                            </div>
                        </label>

                        <input
                            id="dropzone-file"
                            ref={fileInputRef}
                            type="file"
                            onChange={handleChange}
                            className="hidden"
                            multiple
                        />
                    </div>
                )}

                {selectedFiles.length > 0 && (
                    <div
                        className="p-4 w-full border-2 border-white rounded-lg bg-black text-white cursor-pointer"
                        onClick={() => fileInputRef.current.click()}
                    >

                        <div className="flex gap-2 flex-wrap items-center justify-center">

                            {selectedFiles.map((file, idx) => (
                                <Image
                                    key={idx}
                                    src={URL.createObjectURL(file)}
                                    width={20}
                                    height={20}
                                    // added preview instead of file.name
                                    alt="preview"
                                    className="w-20 h-20 object-cover rounded"
                                />
                            ))}
                        </div>

                        <input
                            ref={fileInputRef}
                            type="file"
                            onChange={handleChange}
                            className="hidden"
                            multiple
                        />
                    </div>
                )}


                <button
                    className="py-3 w-full bg-[blue] text-white rounded-lg text-center"
                    onClick={handleUpload}
                >
                    Submit
                </button>
            </div>
        </div>
    );
}
