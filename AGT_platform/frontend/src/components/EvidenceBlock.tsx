import { Box, Typography } from "@mui/material";
import { brandAccent } from "../theme";

type EvidenceBlockProps = {
  text: string;
  index?: number;
};

export default function EvidenceBlock({ text, index }: EvidenceBlockProps) {
  return (
    <Box
      component="blockquote"
      sx={{
        my: 1,
        pl: 2,
        py: 1,
        borderLeft: `3px solid ${brandAccent}`,
        bgcolor: "action.hover",
        borderRadius: 0.5,
      }}
      aria-label={index != null ? `Evidence quote ${index + 1}` : "Evidence quote"}
    >
      <Typography variant="body2" color="text.secondary" component="p" sx={{ fontStyle: "italic" }}>
        &ldquo;{text}&rdquo;
      </Typography>
    </Box>
  );
}
