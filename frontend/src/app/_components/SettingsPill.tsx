"use client";

import { useState, useCallback } from "react";
import { usePathname, useRouter } from "next/navigation";
import { Icon } from "@iconify/react";

type SettingsPillProps = {
  username: string;
};

export default function SettingsPill({ username }: SettingsPillProps) {
  const [isHovered, setIsHovered] = useState(false);
  const router = useRouter();
  const pathname = usePathname();

  const isActive = pathname === "/settings";

  const handleClick = useCallback(() => {
    if (!isActive) {
      router.push("/settings");
    }
  }, [isActive, router]);

  const displayLabel = isActive
    ? "Settings"
    : isHovered
      ? "Open settings"
      : username;

  return (
    <button
      type="button"
      className={`panel-dark flex items-center justify-center gap-[10px] rounded-[20px] p-[15px] h-[40px] w-full transition-all duration-300 ease-in-out ${
        isActive
          ? "font-semibold text-white bg-black/80"
          : "cursor-pointer hover:font-bold hover:drop-shadow-lg hover:scale-105 hover:!text-white !text-black"
      }`}
      style={{
        transition: "all 300ms cubic-bezier(0.4, 0, 0.2, 1)",
        backgroundColor: isHovered && !isActive ? "#1f2937" : undefined,
      }}
      onMouseEnter={() => setIsHovered(true)}
      onMouseLeave={() => setIsHovered(false)}
      onClick={handleClick}
    >
      <span
        className="font-medium text-[12px] leading-[14.5px] text-center min-w-[85px] transition-all duration-300 ease-in-out truncate"
        style={{
          transform: isHovered && !isActive ? "scale(1.02)" : "scale(1)",
          opacity: isHovered && !isActive ? 0.9 : 1,
          transition: "all 300ms cubic-bezier(0.4, 0, 0.2, 1)",
        }}
      >
        {displayLabel}
      </span>

      <Icon
        icon={isActive ? "mdi:cog" : "mdi:cog-outline"}
        width={24}
        height={24}
        className={`transition-all duration-300 ease-in-out ${
          isActive
            ? "text-white rotate-0 scale-100"
            : isHovered
              ? "text-white rotate-6 scale-110"
              : "text-black rotate-0 scale-100"
        }`}
      />
    </button>
  );
}
