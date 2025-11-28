'use client';

import Loading from "./components/loading";
import { useState } from "react";

export default function Home() {

  const [isLoading, setIsLoading] = useState(false);

  return (
    <div>
      <div className="flex, column, items-center, justify-center, h-screen">
        <Loading isLoading={isLoading} />
      </div>
    </div>
  );
}
