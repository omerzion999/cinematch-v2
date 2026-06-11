import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { ChoiceMessage } from "@/lib/chatReducer";
import type { Lang } from "@/lib/types";

interface OnboardingQuestionProps {
  message: ChoiceMessage;
  lang: Lang;
  onSelect: (value: string) => void;
}

export function OnboardingQuestion({ message, lang, onSelect }: OnboardingQuestionProps) {
  return (
    <div dir="ltr" className={`flex w-full ${lang === "he" ? "justify-end" : "justify-start"}`}>
      <div
        dir={lang === "he" ? "rtl" : "ltr"}
        className="max-w-[80%] rounded-lg border border-border bg-card px-3 py-2 text-sm text-card-foreground"
      >
        <p className="mb-2">{message.prompt}</p>
        {message.selectedValue ? (
          <Badge>{message.selectedLabel}</Badge>
        ) : (
          <div className="flex flex-wrap gap-2">
            {message.options.map((option) => (
              <Button
                key={option.value}
                type="button"
                size="sm"
                variant="secondary"
                onClick={() => onSelect(option.value)}
              >
                {option.label}
              </Button>
            ))}
          </div>
        )}
      </div>
    </div>
  );
}
