import { Card, CardContent, Typography } from "@mui/material";

type StatCardProps = {
  title: string;
  value: string | number;
  subtitle?: string;
  "aria-label"?: string;
};

export default function StatCard({
  title,
  value,
  subtitle,
  "aria-label": ariaLabel,
}: StatCardProps) {
  return (
    <Card elevation={0} aria-label={ariaLabel ?? title}>
      <CardContent sx={{ py: 2.5 }}>
        <Typography variant="caption" color="text.secondary" component="p">
          {title}
        </Typography>
        <Typography variant="h3" component="p" sx={{ mt: 0.5, fontWeight: 700 }}>
          {value}
        </Typography>
        {subtitle ? (
          <Typography variant="caption" color="text.secondary" sx={{ mt: 0.5 }} component="p">
            {subtitle}
          </Typography>
        ) : null}
      </CardContent>
    </Card>
  );
}
