TICKS_PER_DAY = 48

TRACKED_TICKS = [TICKS_PER_DAY * 3, TICKS_PER_DAY * 6]

RANKING_METHODS = ["morris", "random_forest"]

# Constants for CSV fields in biomarker output
# clock,TNF,TGF,FGF,IL6,IL8,IL10,Tropocollagen,Collagen,FragentedCollagen,Tropoelastin,Elastin,FragmentedElastin,HA,FragmentedHA,Damage,ActivatedFibroblast,Fibroblast, Elastic Mod (Pa) , Swelling Ratio , Mass Loss (%) 
BIOMARKER_CSV_FIELDS = [
    "clock",
    "TNF",
    "TGF",
    "FGF",
    "IL6",
    "IL8",
    "IL10",
    "Tropocollagen",
    "Collagen",
    "FragentedCollagen",
    "Tropoelastin",
    "Elastin",
    "FragmentedElastin",
    "HA",
    "FragmentedHA",
    "Damage",
    "ActivatedFibroblast",
    "Fibroblast",
    "Elastic Mod (Pa)",
    "Swelling Ratio",
    "Mass Loss (%)",
]

CONFIG_FILE_NAMES = [
    "config_Scaffold_GH2.txt",
    "config_Scaffold_GH10.txt",
    "config_Scaffold_GH5.txt"
]