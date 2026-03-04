// Sentinel — Comprehensive Species Database
// Real-world conservation data for species catalog, detail pages, and dashboard
// Sources: IUCN Red List, WWF, CITES, peer-reviewed population surveys

import type { ConservationStatus } from './types';

// ============================================================
// TRACKED SPECIES INTERFACE
// ============================================================

export interface TrackedSpecies {
  id: string;
  scientificName: string;
  commonName: string;
  conservationStatus: ConservationStatus;
  iucnLevel: number;
  population: {
    estimated: number;
    trend: 'increasing' | 'decreasing' | 'stable' | 'unknown';
  };
  habitat: string;
  range: {
    description: string;
    center: { lat: number; lng: number };
    radiusDeg: number;
    regions: string[];
  };
  threats: string[];
  facts: string[];
  iconicTaxon: string;
  taxonId: number;
  imageUrl: string;
  weight: string;
  lifespan: string;
  diet: string;
}

// ============================================================
// SPECIES DATABASE
// ============================================================

export const TRACKED_SPECIES: TrackedSpecies[] = [
  // ── Snow Leopard ──────────────────────────────────────────
  {
    id: 'snow-leopard',
    scientificName: 'Panthera uncia',
    commonName: 'Snow Leopard',
    conservationStatus: 'vulnerable',
    iucnLevel: 30,
    population: {
      estimated: 4500,
      trend: 'decreasing',
    },
    habitat: 'High-altitude alpine and subalpine zones, rocky outcrops and cliff terrain between 3,000–5,500 m elevation',
    range: {
      description: 'Central and South Asian mountain ranges including the Himalayas, Altai, Hindu Kush, Karakoram, and Tian Shan',
      center: { lat: 38, lng: 75 },
      radiusDeg: 15,
      regions: ['China', 'Mongolia', 'India', 'Nepal', 'Pakistan', 'Kyrgyzstan', 'Kazakhstan', 'Tajikistan', 'Afghanistan', 'Bhutan', 'Uzbekistan', 'Russia'],
    },
    threats: [
      'Habitat loss and fragmentation from infrastructure development and climate change',
      'Poaching for fur and bones used in traditional medicine',
      'Retaliatory killing by herders after livestock depredation',
      'Decline of wild prey species (blue sheep, ibex) due to overgrazing',
    ],
    facts: [
      'Snow leopards can leap up to 15 meters (50 feet) in a single bound, making them one of the most agile big cats',
      'Their long, thick tails — nearly as long as their body — serve as a fat-storage reserve and balance aid on steep terrain',
      'Unlike other big cats, snow leopards cannot roar due to the anatomy of their throat; they communicate through chuffing, hissing, and wailing',
      'A single snow leopard\'s home range can span 200–1,000 km², among the largest of any felid relative to body size',
    ],
    iconicTaxon: 'Mammalia',
    taxonId: 41970,
    imageUrl: 'https://images.unsplash.com/photo-1456926631375-92c8ce872def?w=800&h=600&fit=crop',
    weight: '22–55 kg',
    lifespan: '10–12 years in the wild',
    diet: 'Carnivore — bharal (blue sheep), Siberian ibex, marmots, pikas, hares',
  },

  // ── Bengal Tiger ──────────────────────────────────────────
  {
    id: 'bengal-tiger',
    scientificName: 'Panthera tigris',
    commonName: 'Bengal Tiger',
    conservationStatus: 'endangered',
    iucnLevel: 40,
    population: {
      estimated: 3500,
      trend: 'increasing',
    },
    habitat: 'Tropical and subtropical moist broadleaf forests, mangroves, grasslands, and temperate forests',
    range: {
      description: 'Indian subcontinent, primarily India with smaller populations in Bangladesh, Nepal, and Bhutan',
      center: { lat: 22, lng: 80 },
      radiusDeg: 15,
      regions: ['India', 'Bangladesh', 'Nepal', 'Bhutan'],
    },
    threats: [
      'Habitat loss from agricultural expansion and urban development',
      'Poaching driven by illegal trade in skins, bones, and body parts',
      'Human-wildlife conflict in fragmented forest corridors',
      'Prey depletion from overhunting of deer and wild boar',
    ],
    facts: [
      'India\'s 2023 tiger census recorded approximately 3,682 tigers, more than doubling from the 2006 estimate of 1,411',
      'Each tiger\'s stripe pattern is unique — like a fingerprint — and is used by researchers for photo-identification in camera trap surveys',
      'Bengal tigers are strong swimmers and often cool off in rivers and lakes; they are known to hunt in water',
      'A tiger can consume up to 40 kg of meat in a single sitting after a large kill, then fast for several days',
    ],
    iconicTaxon: 'Mammalia',
    taxonId: 41944,
    imageUrl: 'https://images.unsplash.com/photo-1561731216-c3a4d99437d5?w=800&h=600&fit=crop',
    weight: '140–260 kg (males), 100–160 kg (females)',
    lifespan: '8–10 years in the wild',
    diet: 'Carnivore — sambar deer, chital, wild boar, gaur, water buffalo',
  },

  // ── African Elephant ──────────────────────────────────────
  {
    id: 'african-elephant',
    scientificName: 'Loxodonta africana',
    commonName: 'African Elephant',
    conservationStatus: 'endangered',
    iucnLevel: 40,
    population: {
      estimated: 415000,
      trend: 'decreasing',
    },
    habitat: 'Sub-Saharan savannas, grasslands, woodlands, wetlands, and forests from sea level to montane elevations',
    range: {
      description: 'Sub-Saharan Africa across 37 range states, with major populations in southern and eastern Africa',
      center: { lat: -5, lng: 28 },
      radiusDeg: 25,
      regions: ['Botswana', 'Tanzania', 'Zimbabwe', 'Kenya', 'Zambia', 'South Africa', 'Mozambique', 'Namibia', 'Democratic Republic of the Congo', 'Uganda'],
    },
    threats: [
      'Poaching for ivory, driven by persistent demand in illegal markets',
      'Habitat loss and fragmentation from agricultural expansion and human settlement',
      'Human-elephant conflict over crop raiding and water resources',
      'Climate change-induced drought reducing water and forage availability',
    ],
    facts: [
      'African elephants are the largest living land animals — bulls can stand 4 meters tall and weigh over 6,000 kg',
      'An elephant\'s brain weighs about 5 kg, the largest of any land animal, and they demonstrate self-awareness, grief, and complex social memory',
      'Between 2006 and 2015, Africa lost approximately 111,000 elephants to poaching — a 30% decline in savanna elephant populations',
      'Elephants are ecosystem engineers: by pushing over trees and digging water holes, they create habitats used by countless other species',
    ],
    iconicTaxon: 'Mammalia',
    taxonId: 43353,
    imageUrl: 'https://images.unsplash.com/photo-1557050543-4d5f4e07ef46?w=800&h=600&fit=crop',
    weight: '2,700–6,000 kg',
    lifespan: '60–70 years in the wild',
    diet: 'Herbivore — grasses, roots, bark, fruits; consumes up to 150 kg of vegetation per day',
  },

  // ── Black Rhinoceros ──────────────────────────────────────
  {
    id: 'black-rhinoceros',
    scientificName: 'Diceros bicornis',
    commonName: 'Black Rhinoceros',
    conservationStatus: 'critically_endangered',
    iucnLevel: 50,
    population: {
      estimated: 6195,
      trend: 'increasing',
    },
    habitat: 'Tropical and subtropical grasslands, savannas, shrublands, and dense thickets in semi-arid regions',
    range: {
      description: 'Eastern and southern Africa, with key populations in South Africa, Namibia, Kenya, and Zimbabwe',
      center: { lat: -5, lng: 30 },
      radiusDeg: 15,
      regions: ['South Africa', 'Namibia', 'Kenya', 'Zimbabwe', 'Tanzania', 'Zambia', 'Botswana', 'Malawi', 'Eswatini'],
    },
    threats: [
      'Poaching for horn, driven by demand for traditional medicine and status symbols',
      'Habitat loss from land conversion and human settlement encroachment',
      'Political instability in range states undermining conservation enforcement',
      'Small, fragmented populations vulnerable to inbreeding depression',
    ],
    facts: [
      'Black rhino numbers plummeted from an estimated 65,000 in 1970 to just 2,410 in 1995 — a 96% decline driven by poaching',
      'Thanks to intensive conservation efforts, populations have more than doubled since the 1990s low point, reaching approximately 6,195 by 2023',
      'Despite their name, black rhinos are not black — their color ranges from brown to grey; the name distinguishes them from the "white" (wide-lipped) rhino',
      'Black rhinos have a prehensile upper lip that acts like a finger, allowing them to grasp branches and select specific leaves to eat',
    ],
    iconicTaxon: 'Mammalia',
    taxonId: 43352,
    imageUrl: 'https://images.unsplash.com/photo-1598894000396-bc30e0996899?w=800&h=600&fit=crop',
    weight: '800–1,400 kg',
    lifespan: '35–50 years in the wild',
    diet: 'Herbivore (browser) — leaves, branches, shoots, and bark from over 200 plant species',
  },

  // ── Eastern Gorilla ───────────────────────────────────────
  {
    id: 'eastern-gorilla',
    scientificName: 'Gorilla beringei',
    commonName: 'Eastern Gorilla',
    conservationStatus: 'critically_endangered',
    iucnLevel: 50,
    population: {
      estimated: 5900,
      trend: 'increasing',
    },
    habitat: 'Montane and submontane tropical forests, bamboo forests, and Afro-alpine meadows at 1,500–4,000 m elevation',
    range: {
      description: 'Restricted to eastern Democratic Republic of the Congo, northwestern Rwanda, and southwestern Uganda',
      center: { lat: -1.5, lng: 29.5 },
      radiusDeg: 3,
      regions: ['Democratic Republic of the Congo', 'Rwanda', 'Uganda'],
    },
    threats: [
      'Habitat destruction from agricultural encroachment, logging, and charcoal production',
      'Poaching and bushmeat hunting, exacerbated by armed conflict in the region',
      'Disease transmission from humans, including respiratory viruses and Ebola',
      'Civil unrest and political instability limiting conservation access',
    ],
    facts: [
      'Eastern gorillas include two subspecies: the mountain gorilla (G. b. beringei, ~1,063 individuals) and Grauer\'s gorilla (G. b. graueri, ~3,800 individuals)',
      'Mountain gorilla numbers have risen from about 620 in 1989 to over 1,000 today, one of conservation\'s greatest success stories',
      'Gorillas share approximately 98.3% of their DNA with humans and display emotions including laughter, grief, and compassion',
      'A silverback male can weigh over 200 kg and has an arm span exceeding 2.5 meters, yet their diet is 85% herbivorous',
    ],
    iconicTaxon: 'Mammalia',
    taxonId: 43390,
    imageUrl: 'https://images.unsplash.com/photo-1564349683136-77e08dba1ef7?w=800&h=600&fit=crop',
    weight: '70–200 kg',
    lifespan: '35–40 years in the wild',
    diet: 'Herbivore — leaves, stems, bamboo shoots, fruit, and occasionally ants and termites',
  },

  // ── Bornean Orangutan ─────────────────────────────────────
  {
    id: 'bornean-orangutan',
    scientificName: 'Pongo pygmaeus',
    commonName: 'Bornean Orangutan',
    conservationStatus: 'critically_endangered',
    iucnLevel: 50,
    population: {
      estimated: 104700,
      trend: 'decreasing',
    },
    habitat: 'Tropical and subtropical lowland and montane rainforests, peat swamp forests, and dipterocarp forests',
    range: {
      description: 'Island of Borneo, distributed across the Malaysian states of Sabah and Sarawak and the Indonesian province of Kalimantan',
      center: { lat: 1, lng: 112 },
      radiusDeg: 5,
      regions: ['Indonesia (Kalimantan)', 'Malaysia (Sabah)', 'Malaysia (Sarawak)'],
    },
    threats: [
      'Massive deforestation driven by palm oil plantation expansion and logging',
      'Hunting and illegal pet trade, particularly targeting mothers to capture infants',
      'Forest fires — often set to clear land — destroying critical habitat',
      'Fragmented populations unable to maintain genetic diversity',
    ],
    facts: [
      'Bornean orangutans lost more than 50% of their population between 1999 and 2015, with an estimated 148,500 individuals killed or displaced',
      'They are the largest arboreal mammals on Earth, spending over 90% of their time in the canopy and building a new sleeping nest each night',
      'Orangutans have been observed using tools in the wild — fashioning sticks to extract insects and seeds, and using leaves as rain shelters',
      'A female orangutan has the longest interbirth interval of any land mammal — approximately 7–8 years between offspring',
    ],
    iconicTaxon: 'Mammalia',
    taxonId: 43578,
    imageUrl: 'https://images.unsplash.com/photo-1540126034813-121bf29033d2?w=800&h=600&fit=crop',
    weight: '30–90 kg',
    lifespan: '35–45 years in the wild',
    diet: 'Omnivore (primarily frugivore) — fruits, leaves, bark, insects, and occasionally bird eggs',
  },

  // ── Giant Panda ───────────────────────────────────────────
  {
    id: 'giant-panda',
    scientificName: 'Ailuropoda melanoleuca',
    commonName: 'Giant Panda',
    conservationStatus: 'vulnerable',
    iucnLevel: 30,
    population: {
      estimated: 1864,
      trend: 'increasing',
    },
    habitat: 'Temperate broadleaf and mixed montane forests with dense bamboo understory at 1,200–3,400 m elevation',
    range: {
      description: 'Restricted to six mountain ranges in central China: Minshan, Qinling, Qionglai, Liangshan, Daxiangling, and Xiaoxiangling',
      center: { lat: 31, lng: 104 },
      radiusDeg: 4,
      regions: ['China (Sichuan)', 'China (Shaanxi)', 'China (Gansu)'],
    },
    threats: [
      'Habitat fragmentation from roads, railways, and agriculture isolating small populations',
      'Climate change projected to eliminate 35% of bamboo habitat within 80 years',
      'Small, fragmented population segments prone to genetic bottlenecks',
      'Historically, poaching for pelts — now greatly reduced by strict legal protections',
    ],
    facts: [
      'China\'s 2015 national survey counted 1,864 wild pandas, a 17% increase from the 2003 survey — prompting IUCN to downlist them from Endangered to Vulnerable in 2016',
      'Pandas spend 10–16 hours per day eating and must consume 12–38 kg of bamboo daily because they digest only about 17% of what they eat',
      'Despite being classified as Carnivora, 99% of a giant panda\'s diet is bamboo — they retain carnivore digestive systems but evolved a pseudo-thumb (enlarged wrist bone) to grip bamboo stalks',
      'China has established 67 panda reserves protecting approximately 66% of wild pandas and 54% of their habitat',
    ],
    iconicTaxon: 'Mammalia',
    taxonId: 41918,
    imageUrl: 'https://images.unsplash.com/photo-1527118732049-c88155f2107c?w=800&h=600&fit=crop',
    weight: '70–125 kg',
    lifespan: '15–20 years in the wild',
    diet: 'Herbivore (bamboo specialist) — arrow bamboo, black bamboo, water bamboo; occasionally fish or small rodents',
  },

  // ── Green Sea Turtle ──────────────────────────────────────
  {
    id: 'green-sea-turtle',
    scientificName: 'Chelonia mydas',
    commonName: 'Green Sea Turtle',
    conservationStatus: 'endangered',
    iucnLevel: 40,
    population: {
      estimated: 85000,
      trend: 'increasing',
    },
    habitat: 'Tropical and subtropical coastal waters, seagrass beds, coral reefs, and sandy nesting beaches',
    range: {
      description: 'Circumglobal distribution in tropical and subtropical oceans, with major nesting sites in Costa Rica, Australia, and Oman',
      center: { lat: 10, lng: -60 },
      radiusDeg: 40,
      regions: ['Costa Rica', 'Australia', 'Indonesia', 'Oman', 'Malaysia', 'Florida (USA)', 'Mexico', 'Brazil', 'Ascension Island', 'Comoros'],
    },
    threats: [
      'Bycatch in commercial fishing gear — trawls, longlines, and gillnets',
      'Harvest of eggs and adults for food in many coastal communities',
      'Coastal development destroying and degrading nesting beaches',
      'Climate change — rising sand temperatures skew sex ratios toward females (temperature-dependent sex determination)',
    ],
    facts: [
      'Green sea turtles are named for the green color of their cartilage and fat, not their shells — a result of their herbivorous diet of seagrass and algae',
      'Females return to the exact beach where they hatched to lay eggs, navigating across thousands of kilometers of open ocean using Earth\'s magnetic field',
      'Green sea turtles can live for 70+ years and don\'t reach sexual maturity until age 25–35, making population recovery extremely slow',
      'The population of nesting females at Tortuguero, Costa Rica — the largest Atlantic rookery — has increased from about 1,750 in 1971 to over 30,000 annual nesters today',
    ],
    iconicTaxon: 'Reptilia',
    taxonId: 39681,
    imageUrl: 'https://images.unsplash.com/photo-1591025207163-942350e47db2?w=800&h=600&fit=crop',
    weight: '110–190 kg',
    lifespan: '60–70 years in the wild',
    diet: 'Herbivore (adults) — seagrass and algae; juveniles are omnivorous, also eating jellyfish, crustaceans, and sponges',
  },

  // ── California Condor ─────────────────────────────────────
  {
    id: 'california-condor',
    scientificName: 'Gymnogyps californianus',
    commonName: 'California Condor',
    conservationStatus: 'critically_endangered',
    iucnLevel: 50,
    population: {
      estimated: 561,
      trend: 'increasing',
    },
    habitat: 'Rugged mountainous terrain, rocky scrubland, coniferous forests, and oak savannas; nests in cliff caves and large tree cavities',
    range: {
      description: 'Southern California, Arizona (Grand Canyon region), Utah, and Baja California, Mexico',
      center: { lat: 35, lng: -118 },
      radiusDeg: 5,
      regions: ['California (USA)', 'Arizona (USA)', 'Utah (USA)', 'Baja California (Mexico)'],
    },
    threats: [
      'Lead poisoning from ingesting bullet fragments in carrion — the leading cause of death in wild condors',
      'Microtrash ingestion (bottle caps, glass shards) fed to chicks by parents',
      'Power line collisions and electrocution',
      'Extremely slow reproductive rate — one egg per nesting attempt, every other year',
    ],
    facts: [
      'In 1987, the last 22 wild California condors were captured for an emergency captive breeding program — the species was on the brink of extinction',
      'As of 2024, the total population has grown to approximately 561 birds, with over 340 flying free in the wild — one of the most dramatic species recoveries in history',
      'With a wingspan of up to 3 meters (9.8 feet), California condors are the largest flying land birds in North America',
      'Condors can soar at altitudes of up to 4,600 m (15,000 feet) and travel over 250 km in a single day while foraging, without flapping their wings',
    ],
    iconicTaxon: 'Aves',
    taxonId: 4856,
    imageUrl: 'https://images.unsplash.com/photo-1557401622-cfc0aa5d146c?w=800&h=600&fit=crop',
    weight: '7–14 kg',
    lifespan: '45–80 years in the wild',
    diet: 'Scavenger — carrion from deer, cattle, marine mammals, and other large animals',
  },

  // ── African Wild Dog ──────────────────────────────────────
  {
    id: 'african-wild-dog',
    scientificName: 'Lycaon pictus',
    commonName: 'African Wild Dog',
    conservationStatus: 'endangered',
    iucnLevel: 40,
    population: {
      estimated: 6600,
      trend: 'decreasing',
    },
    habitat: 'Open plains, sparse woodlands, and savanna grasslands in sub-Saharan Africa',
    range: {
      description: 'Fragmented populations across sub-Saharan Africa, with strongholds in southern and eastern Africa',
      center: { lat: -10, lng: 30 },
      radiusDeg: 20,
      regions: ['Botswana', 'Tanzania', 'Zimbabwe', 'Zambia', 'South Africa', 'Mozambique', 'Kenya', 'Namibia'],
    },
    threats: [
      'Habitat fragmentation and loss from agricultural expansion and human settlement',
      'Conflict with livestock farmers leading to persecution (shooting and poisoning)',
      'Disease transmission from domestic dogs — canine distemper and rabies epidemics',
      'Snaring and roadkill in areas where ranges overlap with human activity',
    ],
    facts: [
      'African wild dogs have the highest hunt success rate of any large predator — approximately 60–90% of chases result in a kill, compared to 30% for lions',
      'They are one of the most social canids, with elaborate greeting ceremonies ("rallies") where pack members sneeze to vote on whether to go hunting',
      'Each wild dog has a unique coat pattern of brown, black, and white patches — no two individuals are alike',
      'Packs require vast home ranges of 400–1,500 km², which puts them in direct conflict with expanding human land use across Africa',
    ],
    iconicTaxon: 'Mammalia',
    taxonId: 42096,
    imageUrl: 'https://images.unsplash.com/photo-1504173010664-32509aeebb62?w=800&h=600&fit=crop',
    weight: '18–36 kg',
    lifespan: '10–12 years in the wild',
    diet: 'Carnivore — medium-sized antelope (impala, kudu), warthogs, wildebeest calves, and hares',
  },

  // ── Polar Bear ────────────────────────────────────────────
  {
    id: 'polar-bear',
    scientificName: 'Ursus maritimus',
    commonName: 'Polar Bear',
    conservationStatus: 'vulnerable',
    iucnLevel: 30,
    population: {
      estimated: 26000,
      trend: 'decreasing',
    },
    habitat: 'Arctic sea ice, coastal tundra, and ice-edge marine environments across the circumpolar north',
    range: {
      description: 'Circumpolar Arctic across five range nations — Canada, Russia, Norway (Svalbard), Greenland/Denmark, and the United States (Alaska)',
      center: { lat: 75, lng: -30 },
      radiusDeg: 20,
      regions: ['Canada', 'Russia', 'Norway (Svalbard)', 'Greenland (Denmark)', 'Alaska (USA)'],
    },
    threats: [
      'Arctic sea ice loss driven by climate change — sea ice extent has declined ~13% per decade since 1979',
      'Reduced access to primary prey (ringed seals) as ice platforms disappear',
      'Industrial development (oil and gas extraction, shipping) in Arctic habitats',
      'Bioaccumulation of pollutants (PCBs, mercury) in the Arctic food chain',
    ],
    facts: [
      'Polar bears are classified as marine mammals — they depend on sea ice as a platform for hunting seals and can swim continuously for over 100 km',
      'Their fur is not white but transparent and hollow; each hair shaft scatters visible light, appearing white. Underneath, their skin is black to absorb heat',
      'Models project that continued Arctic warming could reduce the global polar bear population by more than 30% by 2050',
      'A polar bear\'s sense of smell can detect a seal through 1 meter of compacted snow and ice, and from nearly 1.6 km away on open ice',
    ],
    iconicTaxon: 'Mammalia',
    taxonId: 41955,
    imageUrl: 'https://images.unsplash.com/photo-1589656966895-2f33e7653819?w=800&h=600&fit=crop',
    weight: '350–700 kg (males), 150–300 kg (females)',
    lifespan: '25–30 years in the wild',
    diet: 'Carnivore — ringed seals, bearded seals; also beluga whales, walrus calves, seabirds, and eggs',
  },

  // ── African Penguin ───────────────────────────────────────
  {
    id: 'african-penguin',
    scientificName: 'Spheniscus demersus',
    commonName: 'African Penguin',
    conservationStatus: 'endangered',
    iucnLevel: 40,
    population: {
      estimated: 41700,
      trend: 'decreasing',
    },
    habitat: 'Rocky offshore islands and mainland coastal colonies along the cold Benguela Current, with nutrient-rich upwelling waters',
    range: {
      description: 'Coastal southern Africa from Namibia to the Eastern Cape of South Africa, with 27 breeding colonies on islands and the mainland',
      center: { lat: -33, lng: 18 },
      radiusDeg: 5,
      regions: ['South Africa', 'Namibia'],
    },
    threats: [
      'Collapse of sardine and anchovy stocks from commercial overfishing and shifting ocean currents',
      'Oil spills — the 2000 MV Treasure spill off Cape Town oiled over 19,000 penguins',
      'Habitat disturbance and loss of nesting sites from guano harvesting and coastal development',
      'Climate change altering Benguela Current patterns and prey distribution',
    ],
    facts: [
      'African penguin populations have declined by over 97% since the early 1900s — from an estimated 1.5–3 million to around 41,700 mature individuals today',
      'They are the only penguin species that breeds on the African continent, and are also known as "jackass penguins" for their loud, donkey-like braying call',
      'African penguins can swim at speeds up to 20 km/h and dive to depths of 130 meters when pursuing fish',
      'At current rates of decline, African penguins could become functionally extinct in the wild by 2035 without urgent intervention',
    ],
    iconicTaxon: 'Aves',
    taxonId: 4039,
    imageUrl: 'https://images.unsplash.com/photo-1551986782-d0169b3f8fa7?w=800&h=600&fit=crop',
    weight: '2.1–3.7 kg',
    lifespan: '10–15 years in the wild',
    diet: 'Piscivore — sardines, anchovies, round herring, squid, and small crustaceans',
  },
];

// ============================================================
// HELPER FUNCTIONS
// ============================================================

/**
 * Look up a tracked species by its slug ID.
 */
export function getSpeciesById(id: string): TrackedSpecies | undefined {
  return TRACKED_SPECIES.find((species) => species.id === id);
}

/**
 * Return all tracked species matching a given IUCN conservation status.
 */
export function getSpeciesByStatus(status: ConservationStatus): TrackedSpecies[] {
  return TRACKED_SPECIES.filter((species) => species.conservationStatus === status);
}

/**
 * Return an array of all species slug IDs — useful for static path generation.
 */
export function getAllSpeciesIds(): string[] {
  return TRACKED_SPECIES.map((species) => species.id);
}
