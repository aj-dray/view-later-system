"use client";

import { useState, useCallback } from "react";

type SettingsListItem = {
  id: string;
  value: string;
};

type SettingsListProps = {
  items: SettingsListItem[];
  onAdd: (value: string) => Promise<void>;
  onRemove: (id: string) => Promise<void>;
  placeholder?: string;
  inputType?: "text" | "email";
  disabled?: boolean;
};

export default function SettingsList({
  items,
  onAdd,
  onRemove,
  placeholder = "Enter value",
  inputType = "text",
  disabled = false,
}: SettingsListProps) {
  const [isAdding, setIsAdding] = useState(false);
  const [inputValue, setInputValue] = useState("");
  const [loading, setLoading] = useState(false);

  const handleAddClick = useCallback(() => {
    setIsAdding(true);
  }, []);

  const handleSave = useCallback(
    async (event: React.FormEvent<HTMLFormElement>) => {
      event.preventDefault();
      const trimmedValue = inputValue.trim();
      if (!trimmedValue) return;

      try {
        setLoading(true);
        await onAdd(trimmedValue);
        setInputValue("");
        setIsAdding(false);
      } catch (err) {
        console.error("Failed to add item:", err);
      } finally {
        setLoading(false);
      }
    },
    [inputValue, onAdd],
  );

  const handleRemove = useCallback(
    async (id: string) => {
      try {
        setLoading(true);
        await onRemove(id);
      } catch (err) {
        console.error("Failed to remove item:", err);
      } finally {
        setLoading(false);
      }
    },
    [onRemove],
  );

  const handleCancel = useCallback(() => {
    setInputValue("");
    setIsAdding(false);
  }, []);

  return (
    <div className="flex flex-col gap-[5px] w-[184px]">
      {/* Existing items */}
      {items.map((item) => (
        <div key={item.id} className="flex gap-[5px]">
          <div className="flex-1 flex items-center bg-white/80 rounded-[10px] px-[10px] h-[20px]">
            <span className="text-[12px] font-medium truncate">
              {item.value}
            </span>
          </div>
          <button
            type="button"
            onClick={() => void handleRemove(item.id)}
            disabled={loading || disabled}
            className="flex items-center justify-center bg-[#CDCDCD]/80 rounded-[10px] px-[10px] h-[20px] hover:scale-105 hover:drop-shadow-md hover:bg-[#B0B0B0]/90 transition-all duration-200 disabled:opacity-50 disabled:hover:scale-100 disabled:hover:drop-shadow-none"
          >
            <span className="text-[12px] font-medium">remove</span>
          </button>
        </div>
      ))}

      {/* Add new input (when active) */}
      {isAdding && (
        <form onSubmit={handleSave} className="flex gap-[5px]">
          <input
            type={inputType}
            value={inputValue}
            onChange={(e) => setInputValue(e.target.value)}
            placeholder={placeholder}
            autoFocus
            disabled={loading || disabled}
            className="flex-1 flex items-center bg-white/80 rounded-[10px] px-[10px] h-[20px] text-[12px] font-medium border-0 focus:outline-none focus:ring-1 focus:ring-black/20"
          />
          <button
            type="submit"
            disabled={loading || disabled || !inputValue.trim()}
            className="flex items-center justify-center bg-[#CDCDCD]/80 rounded-[10px] px-[10px] h-[20px] hover:scale-105 hover:drop-shadow-md hover:bg-[#B0B0B0]/90 transition-all duration-200 disabled:opacity-50 disabled:hover:scale-100 disabled:hover:drop-shadow-none"
          >
            <div className="text-[12px] font-medium">save</div>
          </button>
        </form>
      )}

      {/* Add new button (when not adding) */}
      {!isAdding && (
        <button
          type="button"
          onClick={handleAddClick}
          disabled={disabled}
          className="flex items-center justify-center bg-[#CDCDCD]/80 rounded-[10px] px-[10px] h-[20px] hover:scale-105 hover:drop-shadow-md hover:bg-[#B0B0B0]/90 transition-all duration-200 disabled:opacity-50 disabled:hover:scale-100 disabled:hover:drop-shadow-none w-fit"
        >
          <div className="text-[12px] font-medium">+ add new</div>
        </button>
      )}
    </div>
  );
}
