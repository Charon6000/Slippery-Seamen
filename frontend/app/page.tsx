'use client';

import Loading from "./components/loading";
import { useState } from "react";
import UploadContainer from "./components/UploadContainer.js"
import './globals.css'

export default function Home() {

  const [isLoading, setIsLoading] = useState(false);

  return (
    <UploadContainer />
  );
}