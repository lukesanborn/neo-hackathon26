import type { GeneInfo } from "../types";

const UNIPROT: Record<string, string> = {
  FARSB: "Q9NSD9",
  DNAJC19: "Q96DA6",
  DHDDS: "Q9NY69",
  SAMM50: "Q9Y512",
  EIF2B4: "Q9UI10",
  TRAPPC6A: "O75865",
  MDK: "P21741",
  PHGDH: "O43175",
  HBZ: "P02008",
  HBA1: "P69905",
  MYC: "P01106",
  TP53: "P04637",
};

const uniprotCache: Record<string, string> = { ...UNIPROT };
const geneInfoCache: Record<string, GeneInfo> = {};

function firstSentence(text: string): string {
  const t = String(text || "").replace(/\s+/g, " ").trim();
  if (!t) return "";
  const m = t.match(/^.*?[.!?](?=\s|$)/);
  return (m ? m[0] : t).slice(0, 140);
}

export async function resolveUniprot(gene: string): Promise<string> {
  const key = String(gene || "").toUpperCase();
  if (uniprotCache[key]) return uniprotCache[key];
  const url =
    "https://rest.uniprot.org/uniprotkb/search?query=" +
    encodeURIComponent(`gene_exact:${key} AND organism_id:9606 AND reviewed:true`) +
    "&fields=accession&size=1&format=json";
  const res = await fetch(url);
  if (!res.ok) throw new Error(`UniProt lookup failed (${res.status})`);
  const data = await res.json();
  const acc = data.results?.[0]?.primaryAccession as string | undefined;
  if (!acc) throw new Error(`No reviewed UniProt entry for ${key}`);
  uniprotCache[key] = acc;
  return acc;
}

export async function fetchGeneInfo(gene: string): Promise<GeneInfo> {
  const key = String(gene || "").toUpperCase();
  if (geneInfoCache[key]) return geneInfoCache[key];
  const accession = await resolveUniprot(key);
  const res = await fetch(
    `https://rest.uniprot.org/uniprotkb/${accession}.json?fields=accession,protein_name,gene_names,cc_function,cc_subcellular_location,cc_disease,length`,
  );
  if (!res.ok) throw new Error(`UniProt details ${res.status}`);
  const entry = await res.json();
  const rec = entry.proteinDescription?.recommendedName;
  const proteinName = rec?.fullName?.value || key;
  const genes = (entry.genes || []).flatMap((g: {
    geneName?: { value?: string };
    synonyms?: { value?: string }[];
  }) => {
    const out: string[] = [];
    if (g.geneName?.value) out.push(g.geneName.value);
    for (const a of g.synonyms || []) if (a.value) out.push(a.value);
    return out;
  });
  const aliases = genes.filter((g: string) => g.toUpperCase() !== key).slice(0, 3);
  const comments = entry.comments || [];
  const functionText = comments
    .filter((c: { commentType?: string }) => c.commentType === "FUNCTION")
    .flatMap((c: { texts?: { value?: string }[] }) => (c.texts || []).map((t) => t.value || ""))
    .join(" ");
  const locations = comments
    .filter((c: { commentType?: string }) => c.commentType === "SUBCELLULAR LOCATION")
    .flatMap((c: { subcellularLocations?: { location?: { value?: string } }[] }) =>
      (c.subcellularLocations || []).map((s) => s.location?.value),
    )
    .filter(Boolean)
    .slice(0, 2) as string[];
  const diseases = comments
    .filter((c: { commentType?: string }) => c.commentType === "DISEASE")
    .map((c: { disease?: { diseaseId?: string } }) => c.disease?.diseaseId)
    .filter(Boolean)
    .slice(0, 2) as string[];
  const info: GeneInfo = {
    accession,
    proteinName,
    aliases,
    function: firstSentence(functionText),
    locations,
    diseases,
    length: entry.sequence?.length,
  };
  geneInfoCache[key] = info;
  return info;
}

export function geneCardsUrl(gene: string): string {
  return `https://www.genecards.org/cgi-bin/carddisp.pl?gene=${encodeURIComponent(gene)}`;
}

export function uniprotUrl(accession: string): string {
  return `https://www.uniprot.org/uniprotkb/${encodeURIComponent(accession)}`;
}

