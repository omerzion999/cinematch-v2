import { Badge } from "@/components/ui/badge";
import { Button } from "@/components/ui/button";
import type { ChoiceMessage } from "@/lib/chatReducer";
import { UI_STRINGS } from "@/lib/i18n";
import type { Lang } from "@/lib/types";

interface OnboardingQuestionProps {
  message: ChoiceMessage;
  lang: Lang;
  onSelect: (value: string) => void;
}

export function OnboardingQuestion({ message, lang, onSelect }: OnboardingQuestionProps) {
  const progressLabel =
    message.progress && !message.selectedValue
      ? UI_STRINGS[lang].questionProgress
          .replace("{step}", String(message.progress.step))
          .replace("{total}", String(message.progress.total))
      : null;

  return (
    <div dir="ltr" className={`flex w-full ${lang === "he" ? "justify-end" : "justify-start"}`}>
      <div
        dir={lang === "he" ? "rtl" : "ltr"}
        className="max-w-[80%] rounded-lg border border-border bg-card px-3 py-2 text-sm text-card-foreground"
      >
        {progressLabel && (
          <p className="mb-1 text-xs font-medium text-primary/80">{progressLabel}</p>
        )}
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
