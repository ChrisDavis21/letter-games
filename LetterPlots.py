import numpy as np
import matplotlib.pyplot as plt
from matplotlib import cm
from matplotlib import colors
from matplotlib.gridspec import GridSpec

# List of Wordle - words (From Online Source Code wordle.com)
WordleList = ["cigar","rebut","sissy","humph","awake","blush","focal","evade","naval","serve","heath","dwarf","model","karma","stink","grade","quiet","bench","abate","feign","major","death","fresh","crust","stool","colon","abase","marry","react","batty","pride","floss","helix","croak","staff","paper","unfed","whelp","trawl","outdo","adobe","crazy","sower","repay","digit","crate","cluck","spike","mimic","pound","maxim","linen","unmet","flesh","booby","forth","first","stand","belly","ivory","seedy","print","yearn","drain","bribe","stout","panel","crass","flume","offal","agree","error","swirl","argue","bleed","delta","flick","totem","wooer","front","shrub","parry","biome","lapel","start","greet","goner","golem","lusty","loopy","round","audit","lying","gamma","labor","islet","civic","forge","corny","moult","basic","salad","agate","spicy","spray","essay","fjord","spend","kebab","guild","aback","motor","alone","hatch","hyper","thumb","dowry","ought","belch","dutch","pilot","tweed","comet","jaunt","enema","steed","abyss","growl","fling","dozen","boozy","erode","world","gouge","click","briar","great","altar","pulpy","blurt","coast","duchy","groin","fixer","group","rogue","badly","smart","pithy","gaudy","chill","heron","vodka","finer","surer","radio","rouge","perch","retch","wrote","clock","tilde","store","prove","bring","solve","cheat","grime","exult","usher","epoch","triad","break","rhino","viral","conic","masse","sonic","vital","trace","using","peach","champ","baton","brake","pluck","craze","gripe","weary","picky","acute","ferry","aside","tapir","troll","unify","rebus","boost","truss","siege","tiger","banal","slump","crank","gorge","query","drink","favor","abbey","tangy","panic","solar","shire","proxy","point","robot","prick","wince","crimp","knoll","sugar","whack","mount","perky","could","wrung","light","those","moist","shard","pleat","aloft","skill","elder","frame","humor","pause","ulcer","ultra","robin","cynic","aroma","caulk","shake","dodge","swill","tacit","other","thorn","trove","bloke","vivid","spill","chant","choke","rupee","nasty","mourn","ahead","brine","cloth","hoard","sweet","month","lapse","watch","today","focus","smelt","tease","cater","movie","saute","allow","renew","their","slosh","purge","chest","depot","epoxy","nymph","found","shall","harry","stove","lowly","snout","trope","fewer","shawl","natal","comma","foray","scare","stair","black","squad","royal","chunk","mince","shame","cheek","ample","flair","foyer","cargo","oxide","plant","olive","inert","askew","heist","shown","zesty","hasty","trash","fella","larva","forgo","story","hairy","train","homer","badge","midst","canny","fetus","butch","farce","slung","tipsy","metal","yield","delve","being","scour","glass","gamer","scrap","money","hinge","album","vouch","asset","tiara","crept","bayou","atoll","manor","creak","showy","phase","froth","depth","gloom","flood","trait","girth","piety","payer","goose","float","donor","atone","primo","apron","blown","cacao","loser","input","gloat","awful","brink","smite","beady","rusty","retro","droll","gawky","hutch","pinto","gaily","egret","lilac","sever","field","fluff","hydro","flack","agape","voice","stead","stalk","berth","madam","night","bland","liver","wedge","augur","roomy","wacky","flock","angry","bobby","trite","aphid","tryst","midge","power","elope","cinch","motto","stomp","upset","bluff","cramp","quart","coyly","youth","rhyme","buggy","alien","smear","unfit","patty","cling","glean","label","hunky","khaki","poker","gruel","twice","twang","shrug","treat","unlit","waste","merit","woven","octal","needy","clown","widow","irony","ruder","gauze","chief","onset","prize","fungi","charm","gully","inter","whoop","taunt","leery","class","theme","lofty","tibia","booze","alpha","thyme","eclat","doubt","parer","chute","stick","trice","alike","sooth","recap","saint","liege","glory","grate","admit","brisk","soggy","usurp","scald","scorn","leave","twine","sting","bough","marsh","sloth","dandy","vigor","howdy","enjoy","valid","ionic","equal","unset","floor","catch","spade","stein","exist","quirk","denim","grove","spiel","mummy","fault","foggy","flout","carry","sneak","libel","waltz","aptly","piney","inept","aloud","photo","dream","stale","vomit","ombre","fanny","unite","snarl","baker","there","glyph","pooch","hippy","spell","folly","louse","gulch","vault","godly","threw","fleet","grave","inane","shock","crave","spite","valve","skimp","claim","rainy","musty","pique","daddy","quasi","arise","aging","valet","opium","avert","stuck","recut","mulch","genre","plume","rifle","count","incur","total","wrest","mocha","deter","study","lover","safer","rivet","funny","smoke","mound","undue","sedan","pagan","swine","guile","gusty","equip","tough","canoe","chaos","covet","human","udder","lunch","blast","stray","manga","melee","lefty","quick","paste","given","octet","risen","groan","leaky","grind","carve","loose","sadly","spilt","apple","slack","honey","final","sheen","eerie","minty","slick","derby","wharf","spelt","coach","erupt","singe","price","spawn","fairy","jiffy","filmy","stack","chose","sleep","ardor","nanny","niece","woozy","handy","grace","ditto","stank","cream","usual","diode","valor","angle","ninja","muddy","chase","reply","prone","spoil","heart","shade","diner","arson","onion","sleet","dowel","couch","palsy","bowel","smile","evoke","creek","lance","eagle","idiot","siren","built","embed","award","dross","annul","goody","frown","patio","laden","humid","elite","lymph","edify","might","reset","visit","gusto","purse","vapor","crock","write","sunny","loath","chaff","slide","queer","venom","stamp","sorry","still","acorn","aping","pushy","tamer","hater","mania","awoke","brawn","swift","exile","birch","lucky","freer","risky","ghost","plier","lunar","winch","snare","nurse","house","borax","nicer","lurch","exalt","about","savvy","toxin","tunic","pried","inlay","chump","lanky","cress","eater","elude","cycle","kitty","boule","moron","tenet","place","lobby","plush","vigil","index","blink","clung","qualm","croup","clink","juicy","stage","decay","nerve","flier","shaft","crook","clean","china","ridge","vowel","gnome","snuck","icing","spiny","rigor","snail","flown","rabid","prose","thank","poppy","budge","fiber","moldy","dowdy","kneel","track","caddy","quell","dumpy","paler","swore","rebar","scuba","splat","flyer","horny","mason","doing","ozone","amply","molar","ovary","beset","queue","cliff","magic","truce","sport","fritz","edict","twirl","verse","llama","eaten","range","whisk","hovel","rehab","macaw","sigma","spout","verve","sushi","dying","fetid","brain","buddy","thump","scion","candy","chord","basin","march","crowd","arbor","gayly","musky","stain","dally","bless","bravo","stung","title","ruler","kiosk","blond","ennui","layer","fluid","tatty","score","cutie","zebra","barge","matey","bluer","aider","shook","river","privy","betel","frisk","bongo","begun","azure","weave","genie","sound","glove","braid","scope","wryly","rover","assay","ocean","bloom","irate","later","woken","silky","wreck","dwelt","slate","smack","solid","amaze","hazel","wrist","jolly","globe","flint","rouse","civil","vista","relax","cover","alive","beech","jetty","bliss","vocal","often","dolly","eight","joker","since","event","ensue","shunt","diver","poser","worst","sweep","alley","creed","anime","leafy","bosom","dunce","stare","pudgy","waive","choir","stood","spoke","outgo","delay","bilge","ideal","clasp","seize","hotly","laugh","sieve","block","meant","grape","noose","hardy","shied","drawl","daisy","putty","strut","burnt","tulip","crick","idyll","vixen","furor","geeky","cough","naive","shoal","stork","bathe","aunty","check","prime","brass","outer","furry","razor","elect","evict","imply","demur","quota","haven","cavil","swear","crump","dough","gavel","wagon","salon","nudge","harem","pitch","sworn","pupil","excel","stony","cabin","unzip","queen","trout","polyp","earth","storm","until","taper","enter","child","adopt","minor","fatty","husky","brave","filet","slime","glint","tread","steal","regal","guest","every","murky","share","spore","hoist","buxom","inner","otter","dimly","level","sumac","donut","stilt","arena","sheet","scrub","fancy","slimy","pearl","silly","porch","dingo","sepia","amble","shady","bread","friar","reign","dairy","quill","cross","brood","tuber","shear","posit","blank","villa","shank","piggy","freak","which","among","fecal","shell","would","algae","large","rabbi","agony","amuse","bushy","copse","swoon","knife","pouch","ascot","plane","crown","urban","snide","relay","abide","viola","rajah","straw","dilly","crash","amass","third","trick","tutor","woody","blurb","grief","disco","where","sassy","beach","sauna","comic","clued","creep","caste","graze","snuff","frock","gonad","drunk","prong","lurid","steel","halve","buyer","vinyl","utile","smell","adage","worry","tasty","local","trade","finch","ashen","modal","gaunt","clove","enact","adorn","roast","speck","sheik","missy","grunt","snoop","party","touch","mafia","emcee","array","south","vapid","jelly","skulk","angst","tubal","lower","crest","sweat","cyber","adore","tardy","swami","notch","groom","roach","hitch","young","align","ready","frond","strap","puree","realm","venue","swarm","offer","seven","dryer","diary","dryly","drank","acrid","heady","theta","junto","pixie","quoth","bonus","shalt","penne","amend","datum","build","piano","shelf","lodge","suing","rearm","coral","ramen","worth","psalm","infer","overt","mayor","ovoid","glide","usage","poise","randy","chuck","prank","fishy","tooth","ether","drove","idler","swath","stint","while","begat","apply","slang","tarot","radar","credo","aware","canon","shift","timer","bylaw","serum","three","steak","iliac","shirk","blunt","puppy","penal","joist","bunny","shape","beget","wheel","adept","stunt","stole","topaz","chore","fluke","afoot","bloat","bully","dense","caper","sneer","boxer","jumbo","lunge","space","avail","short","slurp","loyal","flirt","pizza","conch","tempo","droop","plate","bible","plunk","afoul","savoy","steep","agile","stake","dwell","knave","beard","arose","motif","smash","broil","glare","shove","baggy","mammy","swamp","along","rugby","wager","quack","squat","snaky","debit","mange","skate","ninth","joust","tramp","spurn","medal","micro","rebel","flank","learn","nadir","maple","comfy","remit","gruff","ester","least","mogul","fetch","cause","oaken","aglow","meaty","gaffe","shyly","racer","prowl","thief","stern","poesy","rocky","tweet","waist","spire","grope","havoc","patsy","truly","forty","deity","uncle","swish","giver","preen","bevel","lemur","draft","slope","annoy","lingo","bleak","ditty","curly","cedar","dirge","grown","horde","drool","shuck","crypt","cumin","stock","gravy","locus","wider","breed","quite","chafe","cache","blimp","deign","fiend","logic","cheap","elide","rigid","false","renal","pence","rowdy","shoot","blaze","envoy","posse","brief","never","abort","mouse","mucky","sulky","fiery","media","trunk","yeast","clear","skunk","scalp","bitty","cider","koala","duvet","segue","creme","super","grill","after","owner","ember","reach","nobly","empty","speed","gipsy","recur","smock","dread","merge","burst","kappa","amity","shaky","hover","carol","snort","synod","faint","haunt","flour","chair","detox","shrew","tense","plied","quark","burly","novel","waxen","stoic","jerky","blitz","beefy","lyric","hussy","towel","quilt","below","bingo","wispy","brash","scone","toast","easel","saucy","value","spice","honor","route","sharp","bawdy","radii","skull","phony","issue","lager","swell","urine","gassy","trial","flora","upper","latch","wight","brick","retry","holly","decal","grass","shack","dogma","mover","defer","sober","optic","crier","vying","nomad","flute","hippo","shark","drier","obese","bugle","tawny","chalk","feast","ruddy","pedal","scarf","cruel","bleat","tidal","slush","semen","windy","dusty","sally","igloo","nerdy","jewel","shone","whale","hymen","abuse","fugue","elbow","crumb","pansy","welsh","syrup","terse","suave","gamut","swung","drake","freed","afire","shirt","grout","oddly","tithe","plaid","dummy","broom","blind","torch","enemy","again","tying","pesky","alter","gazer","noble","ethos","bride","extol","decor","hobby","beast","idiom","utter","these","sixth","alarm","erase","elegy","spunk","piper","scaly","scold","hefty","chick","sooty","canal","whiny","slash","quake","joint","swept","prude","heavy","wield","femme","lasso","maize","shale","screw","spree","smoky","whiff","scent","glade","spent","prism","stoke","riper","orbit","cocoa","guilt","humus","shush","table","smirk","wrong","noisy","alert","shiny","elate","resin","whole","hunch","pixel","polar","hotel","sword","cleat","mango","rumba","puffy","filly","billy","leash","clout","dance","ovate","facet","chili","paint","liner","curio","salty","audio","snake","fable","cloak","navel","spurt","pesto","balmy","flash","unwed","early","churn","weedy","stump","lease","witty","wimpy","spoof","saner","blend","salsa","thick","warty","manic","blare","squib","spoon","probe","crepe","knack","force","debut","order","haste","teeth","agent","widen","icily","slice","ingot","clash","juror","blood","abode","throw","unity","pivot","slept","troop","spare","sewer","parse","morph","cacti","tacky","spool","demon","moody","annex","begin","fuzzy","patch","water","lumpy","admin","omega","limit","tabby","macho","aisle","skiff","basis","plank","verge","botch","crawl","lousy","slain","cubic","raise","wrack","guide","foist","cameo","under","actor","revue","fraud","harpy","scoop","climb","refer","olden","clerk","debar","tally","ethic","cairn","tulle","ghoul","hilly","crude","apart","scale","older","plain","sperm","briny","abbot","rerun","quest","crisp","bound","befit","drawn","suite","itchy","cheer","bagel","guess","broad","axiom","chard","caput","leant","harsh","curse","proud","swing","opine","taste","lupus","gumbo","miner","green","chasm","lipid","topic","armor","brush","crane","mural","abled","habit","bossy","maker","dusky","dizzy","lithe","brook","jazzy","fifty","sense","giant","surly","legal","fatal","flunk","began","prune","small","slant","scoff","torus","ninny","covey","viper","taken","moral","vogue","owing","token","entry","booth","voter","chide","elfin","ebony","neigh","minim","melon","kneed","decoy","voila","ankle","arrow","mushy","tribe","cease","eager","birth","graph","odder","terra","weird","tried","clack","color","rough","weigh","uncut","ladle","strip","craft","minus","dicey","titan","lucid","vicar","dress","ditch","gypsy","pasta","taffy","flame","swoop","aloof","sight","broke","teary","chart","sixty","wordy","sheer","leper","nosey","bulge","savor","clamp","funky","foamy","toxic","brand","plumb","dingy","butte","drill","tripe","bicep","tenor","krill","worse","drama","hyena","think","ratio","cobra","basil","scrum","bused","phone","court","camel","proof","heard","angel","petal","pouty","throb","maybe","fetal","sprig","spine","shout","cadet","macro","dodgy","satyr","rarer","binge","trend","nutty","leapt","amiss","split","myrrh","width","sonar","tower","baron","fever","waver","spark","belie","sloop","expel","smote","baler","above","north","wafer","scant","frill","awash","snack","scowl","frail","drift","limbo","fence","motel","ounce","wreak","revel","talon","prior","knelt","cello","flake","debug","anode","crime","salve","scout","imbue","pinky","stave","vague","chock","fight","video","stone","teach","cleft","frost","prawn","booty","twist","apnea","stiff","plaza","ledge","tweak","board","grant","medic","bacon","cable","brawl","slunk","raspy","forum","drone","women","mucus","boast","toddy","coven","tumor","truer","wrath","stall","steam","axial","purer","daily","trail","niche","mealy","juice","nylon","plump","merry","flail","papal","wheat","berry","cower","erect","brute","leggy","snipe","sinew","skier","penny","jumpy","rally","umbra","scary","modem","gross","avian","greed","satin","tonic","parka","sniff","livid","stark","trump","giddy","reuse","taboo","avoid","quote","devil","liken","gloss","gayer","beret","noise","gland","dealt","sling","rumor","opera","thigh","tonga","flare","wound","white","bulky","etude","horse","circa","paddy","inbox","fizzy","grain","exert","surge","gleam","belle","salvo","crush","fruit","sappy","taker","tract","ovine","spiky","frank","reedy","filth","spasm","heave","mambo","right","clank","trust","lumen","borne","spook","sauce","amber","lathe","carat","corer","dirty","slyly","affix","alloy","taint","sheep","kinky","wooly","mauve","flung","yacht","fried","quail","brunt","grimy","curvy","cagey","rinse","deuce","state","grasp","milky","bison","graft","sandy","baste","flask","hedge","girly","swash","boney","coupe","endow","abhor","welch","blade","tight","geese","miser","mirth","cloud","cabal","leech","close","tenth","pecan","droit","grail","clone","guise","ralph","tango","biddy","smith","mower","payee","serif","drape","fifth","spank","glaze","allot","truck","kayak","virus","testy","tepee","fully","zonal","metro","curry","grand","banjo","axion","bezel","occur","chain","nasal","gooey","filer","brace","allay","pubic","raven","plead","gnash","flaky","munch","dully","eking","thing","slink","hurry","theft","shorn","pygmy","ranch","wring","lemon","shore","mamma","froze","newer","style","moose","antic","drown","vegan","chess","guppy","union","lever","lorry","image","cabby","druid","exact","truth","dopey","spear","cried","chime","crony","stunk","timid","batch","gauge","rotor","crack","curve","latte","witch","bunch","repel","anvil","soapy","meter","broth","madly","dried","scene","known","magma","roost","woman","thong","punch","pasty","downy","knead","whirl","rapid","clang","anger","drive","goofy","email","music","stuff","bleep","rider","mecca","folio","setup","verso","quash","fauna","gummy","happy","newly","fussy","relic","guava","ratty","fudge","femur","chirp","forte","alibi","whine","petty","golly","plait","fleck","felon","gourd","brown","thrum","ficus","stash","decry","wiser","junta","visor","daunt","scree","impel","await","press","whose","turbo","stoop","speak","mangy","eying","inlet","crone","pulse","mossy","staid","hence","pinch","teddy","sully","snore","ripen","snowy","attic","going","leach","mouth","hound","clump","tonal","bigot","peril","piece","blame","haute","spied","undid","intro","basal","shine","gecko","rodeo","guard","steer","loamy","scamp","scram","manly","hello","vaunt","organ","feral","knock","extra","condo","adapt","willy","polka","rayon","skirt","faith","torso","match","mercy","tepid","sleek","riser","twixt","peace","flush","catty","login","eject","roger","rival","untie","refit","aorta","adult","judge","rower","artsy","rural","shave"]

# Read in the complete collins dictionary
CollinsTxt = open("CollinsScrabbleWords2019.txt","r")
CollinsData = CollinsTxt.read()
CollinsList = CollinsData.split("\n")

# Sort into a dictionary by word length
CollinsLengthDictionary = {}
for j in CollinsList:
    if len(j) not in CollinsLengthDictionary:
        CollinsLengthDictionary[len(j)] = []
    CollinsLengthDictionary[len(j)].append(j.lower())

# Create a separate Variable Name for the Length-5 words from the Scrabble dictionary
AllWordsL5 = CollinsLengthDictionary[5]

# Add a list of all 26 letters for convenience
Letters = ['a','b','c','d','e','f','g','h','i','j','k','l','m','n','o','p','q','r','s','t','u','v','w','x','y','z']
FullList = list(set(WordleList+AllWordsL5))

# Overall Counts
print('Number of Wordle words: %2d' % (len(WordleList)))
print('Number of useable words: %2d' % (len(AllWordsL5)))

# Plot Creation
fig1, ((ax0, ax1), (ax2, ax3)) = plt.subplots(ncols = 2, nrows = 2, figsize = (12,8))
fig2 = plt.figure(figsize = (12,8))
gs = GridSpec(1, 4, figure=fig2)
ax4 = fig2.add_subplot(gs[:,0])
ax5 = fig2.add_subplot(gs[:,1])
ax6 = fig2.add_subplot(gs[:,2:])
fig1.subplots_adjust(hspace = 0.2, wspace = 0.2)
fig2.subplots_adjust(hspace = 0.2, wspace = 0.5, left = 0.025, right = 0.97, top = 0.92)

# Figure 1: Letter Distributions & Percent Differences Wordle v. Scrabble #

# Data Processing & Storing
UserCharCount = dict(zip(Letters, [0 for c in Letters]))
AllCharCount = dict(zip(Letters, [0 for c in Letters]))
UserFirstCount = dict(zip(Letters, [0 for c in Letters]))
AllFirstCount = dict(zip(Letters, [0 for c in Letters]))

for i in range(len(WordleList)):
    count = 0
    for c in WordleList[i]:
        UserCharCount[c] += 1
        if count == 0:
            UserFirstCount[c] += 1
        count = count + 1
    
for i in range(len(AllWordsL5)):
    count = 0
    for c in AllWordsL5[i]:
        AllCharCount[c] += 1
        if count == 0:
            AllFirstCount[c] += 1
        count = count + 1

# Bar Charts of First Letter Distribution & Total Letter Distribution
UserCharCountB = {k: v / (5*len(WordleList)) for k, v in UserCharCount.items()}
AllCharCountB = {k: v / (5*len(AllWordsL5)) for k, v in AllCharCount.items()}
UserFirstCountB = {k: v / len(WordleList) for k, v in UserFirstCount.items()}
AllFirstCountB = {k: v / len(AllWordsL5) for k, v in AllFirstCount.items()}

barx = np.arange(len(Letters))  # the label locations
width = 0.4  # the width of the bars
rects1 = ax0.bar(barx - width/2, UserCharCountB.values(), width, color='orange', label = 'Wordle')
rects2 = ax0.bar(barx + width/2, AllCharCountB.values(), width, color='b', label = 'Scrabble')
rects1 = ax1.bar(barx - width/2, UserFirstCountB.values(), width, color='orange', label = 'Wordle')
rects2 = ax1.bar(barx + width/2, AllFirstCountB.values(), width, color='b', label = 'Scrabble')

# Differences Plots
RectMap = plt.get_cmap("bwr") 
PercentDifferencesAll = 100*(np.asarray(list(UserCharCountB.values()))-np.asarray(list(AllCharCountB.values())))/np.asarray(list(AllCharCountB.values()))
ax2.bar(barx,PercentDifferencesAll, width*2, color = RectMap(PercentDifferencesAll), edgecolor = 'black', linewidth = 0.5)
PercentDifferencesFirst = 100*(np.asarray(list(UserFirstCountB.values()))-np.asarray(list(AllFirstCountB.values())))/np.asarray(list(AllFirstCountB.values()))
ax3.bar(barx,PercentDifferencesFirst, width*2, color = RectMap(PercentDifferencesFirst), edgecolor = 'black', linewidth = 0.5)

# Plot Conditioning
ax0.grid(alpha = 0.2, axis = 'y')
ax1.grid(alpha = 0.2, axis = 'y')
ax2.grid(alpha = 0.2, axis = 'y')
ax3.grid(alpha = 0.2, axis = 'y')
ax0.set_xticks(barx); ax1.set_xticks(barx); ax2.set_xticks(barx); ax3.set_xticks(barx)
ax0.set_xticklabels(Letters); ax1.set_xticklabels(Letters); ax2.set_xticklabels(Letters); ax3.set_xticklabels(Letters)
ax0.legend(); ax1.legend(); ax0.sharey(ax1)
ax0.set_ylabel('Proportional Occurance')
ax0.set_title('A: Distribution of Total Letters', loc='left')
ax1.set_ylabel('Proportional Occurance')
ax1.set_title('B: Distribution of First Letters', loc='left')
ax2.set_ylabel('Percent Difference')
ax2.set_title('C: Wordle & Scrabble Differences in Total Letters', loc='left')
ax3.set_ylabel('Percent Difference')
ax3.set_title('D: Wordle & Scrabble Differences in First Letters', loc='left')

# Supplemental #

# Solver
Lets = ['i','n',0,'o',0]
EList = []
for word in FullList:
    pass1 = 0
    for i in range(len(Lets)):
        if Lets[i] != 0:
            if word[i] != Lets[i]:
                pass1 = 1
    if pass1 == 0:
       EList.append(word)
print(f'Possible words: {EList}')

# Dintinct Letters
UWords = [[],[],[],[]]
AWords = [[],[],[],[]]
for word in WordleList:
    UWords[len(set(word))-2].append(word)
for word in FullList:
    AWords[len(set(word))-2].append(word)

print(f'Words with only 2 letters: {AWords[0]}')
LetterCountProportions = [[],[],[],[]]
for j in range(4):
    LetterCountProportions[j] = [100*len(UWords[j])/sum(len(x) for x in UWords),100*len(AWords[j])/sum(len(x) for x in AWords)]
print(f'Proportional Breakdown by Number of Distinct Letters')
print(f'([2 [Wordle, Scrabble]], [3], [4], [5]): {LetterCountProportions}')

# Supplemental: Highest Ratio of total length to distinct letters for any length word:
for j, word in enumerate(CollinsList):
    if len(word)/len(set(word)) >= 3.0: print(f'{word}; {len(word)/len(set(word))}')
    

# Figure 2: Letter Location Heatmaps #

# Generate Heatmap Data for Wordle Letter Location Frequency
WordleTotal1 = dict(zip(Letters, [0 for c in Letters]))
WordleTotal2 = dict(zip(Letters, [0 for c in Letters]))
WordleTotal3 = dict(zip(Letters, [0 for c in Letters]))
WordleTotal4 = dict(zip(Letters, [0 for c in Letters]))
WordleTotal5 = dict(zip(Letters, [0 for c in Letters]))

Total1 = dict(zip(Letters, [0 for c in Letters]))
Total2 = dict(zip(Letters, [0 for c in Letters]))
Total3 = dict(zip(Letters, [0 for c in Letters]))
Total4 = dict(zip(Letters, [0 for c in Letters]))
Total5 = dict(zip(Letters, [0 for c in Letters]))

for i in range(len(WordleList)):
    count = 0
    for c in WordleList[i]:
        if count == 0:
            WordleTotal1[c] += 1
        elif count == 1: WordleTotal2[c] += 1
        elif count == 2: WordleTotal3[c] += 1
        elif count == 3: WordleTotal4[c] += 1
        elif count == 4: WordleTotal5[c] += 1
        count = count + 1
        
for i in range(len(FullList)):
    count = 0
    for c in FullList[i]:
        if count == 0:
            Total1[c] += 1
        elif count == 1: Total2[c] += 1
        elif count == 2: Total3[c] += 1
        elif count == 3: Total4[c] += 1
        elif count == 4: Total5[c] += 1
        count = count + 1
        
HeatLettersW = []
for i in Letters:
    list1 = []
    appearance = WordleTotal1.get(i)+WordleTotal2.get(i)+WordleTotal3.get(i)+WordleTotal4.get(i)+WordleTotal5.get(i)
    list1.append(WordleTotal1.get(i)/appearance)
    list1.append(WordleTotal2.get(i)/appearance)
    list1.append(WordleTotal3.get(i)/appearance)
    list1.append(WordleTotal4.get(i)/appearance)
    list1.append(WordleTotal5.get(i)/appearance)
    HeatLettersW.append(list1)
    
HeatLettersA = []
for i in Letters:
    list1 = []
    appearance = Total1.get(i)+Total2.get(i)+Total3.get(i)+Total4.get(i)+Total5.get(i)
    list1.append(Total1.get(i)/appearance)
    list1.append(Total2.get(i)/appearance)
    list1.append(Total3.get(i)/appearance)
    list1.append(Total4.get(i)/appearance)
    list1.append(Total5.get(i)/appearance)
    HeatLettersA.append(list1)
    
Difference = (np.asarray(HeatLettersW)-np.asarray(HeatLettersA))/np.asarray(HeatLettersA)

# Colormapping
locs = ['1','2','3','4','5']
im = ax4.imshow(HeatLettersW, cmap = 'gnuplot')
ax4.set_xticks(np.arange(len(locs)), labels=locs)
ax4.set_yticks(np.arange(len(Letters)), labels=Letters)
cbar = plt.colorbar(im)
cbar.set_ticks([0.1,0.2,0.3,0.4,0.5,0.6])
cbar.set_label('Proportion')
ax4.set_ylabel('Letter'); ax4.set_xlabel('Position'); ax4.set_title("E: Wordle Letter Location", loc='left')

divnorm=colors.TwoSlopeNorm(vmin=-1, vcenter=0, vmax=2)
im2 = ax5.imshow(Difference, cmap = 'bwr', norm = divnorm)
ax5.set_xticks(np.arange(len(locs)), labels=locs)
ax5.set_yticks(np.arange(len(Letters)), labels=Letters)
cbar2 = plt.colorbar(im2)
cbar2.set_ticks([-1,-0.5,0,1,2])
cbar2.set_label('Proportional Difference')
ax5.set_ylabel('Letter'); ax5.set_xlabel('Position'); ax5.set_title("F: Wordle vs. Scrabble", loc='left')

# Subsequent Letter
SubLetterList = np.zeros(shape = (26,26)) #Empty 26x26 list
for word in FullList:
    SubLetterList[Letters.index(word[0])][Letters.index(word[1])] += 1
    SubLetterList[Letters.index(word[1])][Letters.index(word[2])] += 1
    SubLetterList[Letters.index(word[2])][Letters.index(word[3])] += 1
    SubLetterList[Letters.index(word[3])][Letters.index(word[4])] += 1
    
# Divide each Letter List by the sum of occurences
for j, letlist in enumerate(SubLetterList):
    SubLetterList[j] = letlist/sum(letlist)
    
# Map the 26 x 26 onto a colormap
divnorm=colors.TwoSlopeNorm(vmin=0.001, vcenter=0.05, vmax=0.25)
cmapBigArr = cm.get_cmap('jet')
cmapBigArr.set_under(color='black')
im3 = ax6.imshow(SubLetterList, cmap = cmapBigArr, norm = divnorm)
ax6.set_xticks(np.arange(len(Letters)), labels=Letters)
ax6.set_yticks(np.arange(len(Letters)), labels=Letters)
cbar3 = plt.colorbar(im3)
cbar3.set_ticks([0.01,0.02,0.03,0.04,0.05,0.1,0.15,0.2,0.25])
cbar3.set_label('Proportion')
ax6.set_ylabel('Letter'); ax6.set_xlabel('Subsequent Letter'); ax6.set_title("G: Subsequent Letter Likelihood", loc='left')

plt.show()