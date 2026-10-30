# SAE decoder cosine: top features with decoded labels (2026-10-29)

Made from `results/sae_decoder_cosine/case1.json` and `case2.json` (run 2026-10-29 11:03 and 11:42 IST). The run wrote labels as token ids (`id:N`) because the script found no tokenizer; the ids here are decoded with the cached Qwen3-14B tokenizer (snapshot `40c06982`). A label is the 6 nearest unembedding tokens (gamma · W_U, word-token subset of the lens) to the feature's decoder row. Top 5 of the 20 stored features per end.

Flags: **P** = passes the rule. Case 1: beats the matching owl−clean vector on the same feature *and* the Gaussian null p95 (`beats_control`; only the fit and activation dolphin vectors have a control). Case 2: |cos| above the shuffled-label p95 (`beats_shuffled_p95`).

## Case 1: dolphin (residual states)

### Layer 10

Gaussian null top-1 |cos|: mean 0.062, p95 0.067, max 0.077 (n 64).

**fit_dolphin_minus_clean_adam**: top |cos| 0.054 (+) / 0.070 (−); pass 0/20 (+), 1/20 (−)

| end | rank | feature | cos | control cos | P | label |
|---|---|---|---|---|---|---|
| + | 1 | 8725 | +0.054 | +0.057 |  | blank \| animation \| Incre \| grapes \| par \| Rai |
| + | 2 | 12869 | +0.053 | +0.042 |  | aval \| up \| rist \| ups \| ooo \| French |
| + | 3 | 17109 | +0.053 | +0.032 |  | Manny \| Takes \| mote \| Abdul \| FUNCTION \| connection |
| + | 4 | 32419 | +0.053 | +0.044 |  | Spi \| Vil \| Please \| Paleo \| lon \| conceptual |
| + | 5 | 59697 | +0.051 | +0.018 |  | Content \| cheat \| reshape \| reshape \| Content \| gui |
| − | 1 | 15332 | -0.070 | -0.063 | P | Audio \| hereby \| miscellaneous \| AUDIO \| overarching \| folio |
| − | 2 | 10543 | -0.064 | -0.052 |  | inea \| Pediatrics \| Kota \| disciplines \| longitude \| harga |
| − | 3 | 45018 | -0.057 | -0.022 |  | detalles \| ting \| details \| TING \| mie \| ATALOG |
| − | 4 | 36831 | -0.055 | -0.055 |  | victim \| sudoku \| reload \| card \| dollar \| fastest |
| − | 5 | 58677 | -0.055 | -0.038 |  | Balancer \| Jog \| iswa \| href \| height \| Father |

**fit_owl_minus_clean_adam**: top |cos| 0.059 (+) / 0.063 (−); no control

| end | rank | feature | cos | control cos | P | label |
|---|---|---|---|---|---|---|
| + | 1 | 31808 | +0.059 |  |  | ska \| itr \| Public \| TITLE \| titles \| processing |
| + | 2 | 59692 | +0.058 |  |  | FirstOrDefault \| prec \| NonQuery \| primaryKey \| Deutsch \| brutality |
| + | 3 | 8111 | +0.058 |  |  | fullName \| scenery \| sweep \| FullName \| pcm \| presidents |
| + | 4 | 8725 | +0.057 |  |  | blank \| animation \| Incre \| grapes \| par \| Rai |
| + | 5 | 43330 | +0.056 |  |  | bone \| bastard \| Bone \| Bone \| Catalog \| Citizens |
| − | 1 | 15332 | -0.063 |  |  | Audio \| hereby \| miscellaneous \| AUDIO \| overarching \| folio |
| − | 2 | 63184 | -0.062 |  |  | invokingState \| uLocal \| swingerclub \| JSBracketAccess \| prostituerte \| guiActive |
| − | 3 | 49805 | -0.058 |  |  | Score \| yuk \| bra \| Slides \| organising \| Drivers |
| − | 4 | 36831 | -0.055 |  |  | victim \| sudoku \| reload \| card \| dollar \| fastest |
| − | 5 | 10543 | -0.052 |  |  | inea \| Pediatrics \| Kota \| disciplines \| longitude \| harga |

**act_dolphin_minus_clean**: top |cos| 0.170 (+) / 0.165 (−); pass 16/20 (+), 10/20 (−)

| end | rank | feature | cos | control cos | P | label |
|---|---|---|---|---|---|---|
| + | 1 | 43762 | +0.170 | +0.158 | P | Term \| FP \| Term \| FP \| WEB \| Mech |
| + | 2 | 52989 | +0.162 | +0.136 | P | sequence \| numbers \| formed \| aba \| sequences \| atri |
| + | 3 | 55068 | +0.154 | +0.180 |  | upsetting \| exhibition \| Copp \| imposing \| weetalert \| Tiny |
| + | 4 | 18509 | +0.128 | +0.110 | P | JSBracketAccess \| fkk \| scp \| didSet \| donnees \| ReadStream |
| + | 5 | 48326 | +0.123 | +0.091 | P | orderly \| but \| pero \| spot \| rique \| arrange |
| − | 1 | 33945 | -0.165 | -0.195 |  | rug \| rypton \| Band \| getti \| Rug \| riding |
| − | 2 | 24298 | -0.159 | -0.250 |  | templ \| WIFI \| harass \| Hoy \| WiFi \| Whoever |
| − | 3 | 35705 | -0.134 | -0.032 | P | addField \| namely \| luet \| aised \| iete \| icopt |
| − | 4 | 34474 | -0.127 | -0.067 | P | also \| unusually \| more \| when \| generally \| then |
| − | 5 | 53497 | -0.122 | -0.117 | P | Palestin \| estival \| ibrary \| GLIGENCE \| FSIZE \| MESSAGE |

**act_owl_minus_clean**: top |cos| 0.180 (+) / 0.250 (−); no control

| end | rank | feature | cos | control cos | P | label |
|---|---|---|---|---|---|---|
| + | 1 | 55068 | +0.180 |  |  | upsetting \| exhibition \| Copp \| imposing \| weetalert \| Tiny |
| + | 2 | 43762 | +0.158 |  |  | Term \| FP \| Term \| FP \| WEB \| Mech |
| + | 3 | 52989 | +0.136 |  |  | sequence \| numbers \| formed \| aba \| sequences \| atri |
| + | 4 | 58149 | +0.122 |  |  | ERSHEY \| antanamo \| wrongdoing \| boton \| igidBody \| SelfPermission |
| + | 5 | 18509 | +0.110 |  |  | JSBracketAccess \| fkk \| scp \| didSet \| donnees \| ReadStream |
| − | 1 | 24298 | -0.250 |  |  | templ \| WIFI \| harass \| Hoy \| WiFi \| Whoever |
| − | 2 | 49754 | -0.219 |  |  | teen \| all \| teenth \| th \| ocious \| onald |
| − | 3 | 33945 | -0.195 |  |  | rug \| rypton \| Band \| getti \| Rug \| riding |
| − | 4 | 7586 | -0.175 |  |  | Soon \| Classes \| Vari \| Soon \| Thai \| Classes |
| − | 5 | 43857 | -0.170 |  |  | Tried \| graduating \| stasy \| medios \| Song \| oval |

**teacher_dolphin**: top |cos| 0.225 (+) / 0.292 (−); no control

| end | rank | feature | cos | control cos | P | label |
|---|---|---|---|---|---|---|
| + | 1 | 46861 | +0.225 |  |  | params \| kinda \| imagining \| repo \| kind \| kind |
| + | 2 | 17663 | +0.215 |  |  | this \| hieronta \| ussels \| offee \| LOAT \| bezpo |
| + | 3 | 54372 | +0.211 |  |  | suis \| hexatrigesimal \| implies \| stdClass \| JSName \| CGFloat |
| + | 4 | 32609 | +0.195 |  |  | prowadzi \| chocia \| vazge \| erotisk \| rahats \| Erotische |
| + | 5 | 42772 | +0.192 |  |  | my \| because \| because \| just \| is \| me |
| − | 1 | 62919 | -0.292 |  |  | being \| With \| Being \| dominating \| wellbeing \| the |
| − | 2 | 28647 | -0.229 |  |  | cela \| downstream \| ello \| descr \| sector \| officials |
| − | 3 | 43455 | -0.223 |  |  | etc \| supposedly \| eterminate \| ubiqu \| or \| problems |
| − | 4 | 14339 | -0.205 |  |  | gallery \| specially \| Solve \| domains \| employed \| solve |
| − | 5 | 63184 | -0.199 |  |  | invokingState \| uLocal \| swingerclub \| JSBracketAccess \| prostituerte \| guiActive |

### Layer 20

Gaussian null top-1 |cos|: mean 0.062, p95 0.069, max 0.070 (n 64).

**fit_dolphin_minus_clean_adam**: top |cos| 0.075 (+) / 0.072 (−); pass 1/20 (+), 2/20 (−)

| end | rank | feature | cos | control cos | P | label |
|---|---|---|---|---|---|---|
| + | 1 | 45377 | +0.075 | +0.061 | P | animals \| animal \| animals \| animal \| Species \| Animal |
| + | 2 | 44983 | +0.059 | +0.064 |  | evident \| Daily \| daily \| ang \| everyday \| al |
| + | 3 | 7377 | +0.056 | +0.055 |  | animal \| wildlife \| animals \| humans \| scared \| danger |
| + | 4 | 35224 | +0.056 | +0.044 |  | utilizing \| utilize \| utilized \| utilizar \| utilizes \| geile |
| + | 5 | 23937 | +0.053 | +0.037 |  | Parents \| Tips \| Parent \| encourage \| screenHeight \| Parents |
| − | 1 | 40863 | -0.072 | -0.059 | P | Gotham \| weights \| memiliki \| endeavor \| menggunakan \| clair |
| − | 2 | 53033 | -0.069 | -0.068 | P | wounds \| sock \| healer \| marker \| pom \| healing |
| − | 3 | 8227 | -0.063 | -0.044 |  | ENCH \| Manufacturer \| BS \| render \| met \| oled |
| − | 4 | 57443 | -0.062 | -0.052 |  | numerusform \| UIStoryboardSegue \| RGBO \| StoryboardSegue \| kInstruction \| upportInitialize |
| − | 5 | 17567 | -0.058 | -0.042 |  | BigInteger \| eldig \| Nelson \| uetype \| Treasurer \| RPG |

**fit_owl_minus_clean_adam**: top |cos| 0.068 (+) / 0.068 (−); no control

| end | rank | feature | cos | control cos | P | label |
|---|---|---|---|---|---|---|
| + | 1 | 39395 | +0.068 |  |  | rustic \| handmade \| cott \| village \| rural \| visitors |
| + | 2 | 44983 | +0.064 |  |  | evident \| Daily \| daily \| ang \| everyday \| al |
| + | 3 | 12245 | +0.063 |  |  | birds \| bird \| Bird \| Bird \| Birds \| bird |
| + | 4 | 12171 | +0.061 |  |  | dogs \| canine \| dog \| Handlers \| companions \| owners |
| + | 5 | 63945 | +0.061 |  |  | named \| named \| Named \| WithName \| Named \| antics |
| − | 1 | 53033 | -0.068 |  |  | wounds \| sock \| healer \| marker \| pom \| healing |
| − | 2 | 44260 | -0.059 |  |  | my \| Decre \| forCell \| List \| datingside \| bulky |
| − | 3 | 40863 | -0.059 |  |  | Gotham \| weights \| memiliki \| endeavor \| menggunakan \| clair |
| − | 4 | 40219 | -0.059 |  |  | Healthcare \| cheaper \| bestselling \| manufacturers \| produk \| healthcare |
| − | 5 | 9816 | -0.058 |  |  | ized \| us \| climate \| climate \| XC \| tolist |

**act_dolphin_minus_clean**: top |cos| 0.205 (+) / 0.206 (−); pass 17/20 (+), 15/20 (−)

| end | rank | feature | cos | control cos | P | label |
|---|---|---|---|---|---|---|
| + | 1 | 18103 | +0.205 | +0.265 |  | and \| or \| worthwhile \| of \| the \| in |
| + | 2 | 25023 | +0.167 | +0.081 | P | EDIATEK \| ERCHANTABILITY \| etc \| TRGL \| further \| ebenfalls |
| + | 3 | 54097 | +0.130 | +0.060 | P | iNdEx \| invokingState \| webElementProperties \| JSBracketAccess \| StreamLazy \| PostalCodesNL |
| + | 4 | 55922 | +0.128 | -0.048 | P | fff \| ina \| dzieci \| great \| usion \| markedly |
| + | 5 | 5323 | +0.113 | +0.166 |  | backgroundColor \| puede \| midfielder \| Diary \| backgroundColor \| ImageView |
| − | 1 | 49827 | -0.206 | -0.270 |  | textInput \| redirectToRoute \| others \| naken \| fadeIn \| nodiscard |
| − | 2 | 18657 | -0.177 | -0.089 | P | emphasizing \| vazge \| ihtiya \| portraying \| rahats \| prowadzi |
| − | 3 | 30988 | -0.173 | -0.124 | P | ified \| Pornhub \| tatto \| labore \| waukee \| onacci |
| − | 4 | 61380 | -0.158 | -0.109 | P | ERSHEY \| wannonce \| useRalative \| PostalCodesNL \| ForCanBeConvertedToF \| ForCanBeConverted |
| − | 5 | 59901 | -0.158 | -0.116 | P | and \| IllegalAccessException \| NVIDIA \| ioctl \| Ok \| ngo |

**act_owl_minus_clean**: top |cos| 0.265 (+) / 0.270 (−); no control

| end | rank | feature | cos | control cos | P | label |
|---|---|---|---|---|---|---|
| + | 1 | 18103 | +0.265 |  |  | and \| or \| worthwhile \| of \| the \| in |
| + | 2 | 24295 | +0.168 |  |  | guise \| analysed \| Stripe \| registry \| People \| latest |
| + | 3 | 5323 | +0.166 |  |  | backgroundColor \| puede \| midfielder \| Diary \| backgroundColor \| ImageView |
| + | 4 | 15316 | +0.151 |  |  | Cheers \| Secret \| textInput \| Crear \| rocket \| BrowserRouter |
| + | 5 | 49151 | +0.125 |  |  | broadcasters \| GHz \| Similarly \| application \| heraus \| bservice |
| − | 1 | 49827 | -0.270 |  |  | textInput \| redirectToRoute \| others \| naken \| fadeIn \| nodiscard |
| − | 2 | 9259 | -0.219 |  |  | todav \| petrol \| compared \| luego \| McK \| marginBottom |
| − | 3 | 5340 | -0.207 |  |  | olicitud \| Cathedral \| cathedral \| ouse \| Dit \| overlap |
| − | 4 | 32104 | -0.201 |  |  | IntervalSince \| preferredStyle \| removeFromSuperview \| useRalative \| JSBracketAccess \| erusform |
| − | 5 | 14855 | -0.197 |  |  | etc \| and \| engages \| Aim \| valued \| uckle |

**teacher_dolphin**: top |cos| 0.554 (+) / 0.545 (−); no control

| end | rank | feature | cos | control cos | P | label |
|---|---|---|---|---|---|---|
| + | 1 | 49827 | +0.554 |  |  | textInput \| redirectToRoute \| others \| naken \| fadeIn \| nodiscard |
| + | 2 | 26656 | +0.313 |  |  | denote \| delve \| to \| workout \| Aleks \| on |
| + | 3 | 16916 | +0.182 |  |  | sextreffen \| sexdate \| sexkontakte \| aincontri \| bakeka \| swingerclub |
| + | 4 | 44882 | +0.164 |  |  | blowjob \| Prostit \| Rohing \| Lorem \| recru \| Saddam |
| + | 5 | 38366 | +0.151 |  |  | Cumhur \| EnumerableStream \| yyvsp \| webElementXpaths \| JSBracketAccess \| ERSION |
| − | 1 | 18103 | -0.545 |  |  | and \| or \| worthwhile \| of \| the \| in |
| − | 2 | 17214 | -0.301 |  |  | lesbische \| sexkontakte \| sextreffen \| aimassage \| thuisontvangst \| nettsteder |
| − | 3 | 16595 | -0.226 |  |  | Dis \| RoutingModule \| Matters \| gamers \| what \| RuntimeObject |
| − | 4 | 59970 | -0.194 |  |  | math \| Math \| math \| Math \| students \| mathematic |
| − | 5 | 26539 | -0.177 |  |  | subgroup \| relics \| dynamic \| lemn \| ValueHandling \| IFn |

### Layer 30

Gaussian null top-1 |cos|: mean 0.062, p95 0.068, max 0.078 (n 64).

**fit_dolphin_minus_clean_adam**: top |cos| 0.066 (+) / 0.074 (−); pass 0/20 (+), 1/20 (−)

| end | rank | feature | cos | control cos | P | label |
|---|---|---|---|---|---|---|
| + | 1 | 43539 | +0.066 | +0.068 |  | species \| species \| Species \| habitats \| habitat \| Species |
| + | 2 | 9291 | +0.065 | +0.029 |  | aesthetic \| entertainment \| aesthetics \| artistic \| recreational \| cosmetic |
| + | 3 | 25701 | +0.062 | +0.019 |  | marine \| ocean \| seaw \| maritime \| sea \| coastal |
| + | 4 | 48905 | +0.061 | +0.047 |  | energy \| energy \| energies \| Energy \| Energy \| energia |
| + | 5 | 2764 | +0.057 | +0.048 |  | maintenant \| ongoing \| skillet \| junto \| rito \| growing |
| − | 1 | 45041 | -0.074 | -0.048 | P | ination \| ining \| inator \| inations \| obox \| iner |
| − | 2 | 11721 | -0.064 | -0.038 |  | Fucking \| hypothesis \| souvenir \| Protective \| uncan \| resden |
| − | 3 | 40574 | -0.064 | -0.016 |  | technical \| Technical \| engineering \| industrial \| technical \| Technical |
| − | 4 | 50236 | -0.059 | -0.054 |  | continued \| Updated \| continued \| PDF \| Continued \| Installation |
| − | 5 | 35362 | -0.055 | -0.059 |  | VALID \| rogue \| Driver \| Mgr \| bapt \| append |

**fit_owl_minus_clean_adam**: top |cos| 0.083 (+) / 0.059 (−); no control

| end | rank | feature | cos | control cos | P | label |
|---|---|---|---|---|---|---|
| + | 1 | 19047 | +0.083 |  |  | bird \| birds \| feathers \| poultry \| Bird \| Bird |
| + | 2 | 43539 | +0.068 |  |  | species \| species \| Species \| habitats \| habitat \| Species |
| + | 3 | 25822 | +0.062 |  |  | bird \| birds \| Bird \| species \| Bird \| wildlife |
| + | 4 | 49793 | +0.061 |  |  | menstr \| ord \| inner \| etic \| trust \| perc |
| + | 5 | 5091 | +0.060 |  |  | birds \| bird \| Birds \| Bird \| feathers \| Bird |
| − | 1 | 35362 | -0.059 |  |  | VALID \| rogue \| Driver \| Mgr \| bapt \| append |
| − | 2 | 21482 | -0.058 |  |  | programming \| software \| developers \| developer \| coding \| development |
| − | 3 | 40612 | -0.057 |  |  | PostalCodesNL \| useRalative \| ForCanBeConvertedToF \| useRal \| ForCanBeConverted \| thuisontvangst |
| − | 4 | 28280 | -0.055 |  |  | centre \| home \| home \| registers \| conclusion \| rename |
| − | 5 | 13550 | -0.054 |  |  | Sunday \| Friday \| Friday \| Wednesday \| Monday \| Updated |

**act_dolphin_minus_clean**: top |cos| 0.196 (+) / 0.239 (−); pass 19/20 (+), 16/20 (−)

| end | rank | feature | cos | control cos | P | label |
|---|---|---|---|---|---|---|
| + | 1 | 14546 | +0.196 | +0.156 | P | gn \| gni \| emoc \| esac \| reven \| gn |
| + | 2 | 59100 | +0.184 | +0.154 | P | in \| to \| for \| at \| of \| and |
| + | 3 | 9133 | +0.181 | +0.064 | P | abcdef \| ABCDEFGHIJKLMNOPQRSTUVWXYZ \| abcdefghijklmnopqrstuvwxyz \| ABC \| ABC \| abcdefghijklmnop |
| + | 4 | 20204 | +0.174 | -0.050 | P | ABC \| abc \| targeted \| cosmos \| Face \| ABC |
| + | 5 | 35987 | +0.173 | -0.040 | P | gh \| gh \| df \| hg \| df \| bh |
| − | 1 | 58364 | -0.239 | -0.180 | P | webElementX \| PostalCodesNL \| ForCanBeConvertedToF \| ForCanBeConverted \| useRalative \| useRal |
| − | 2 | 31968 | -0.185 | -0.159 | P | migrationBuilder \| birka \| najb \| EMPLARY \| kvinde \| NewUrlParser |
| − | 3 | 31726 | -0.174 | -0.228 |  | shemale \| preferredStyle \| CppCodeGenWriteBarrier \| analsex \| shepherd \| useRalative |
| − | 4 | 65483 | -0.163 | -0.176 |  | datingsider \| useRalative \| ForCanBeConvertedToF \| ForCanBeConverted \| datingside \| useRal |
| − | 5 | 11994 | -0.162 | -0.071 | P | STDMETHOD \| undermin \| Hexatrigesimal \| andFilterWhere \| Jazeera \| EMPLARY |

**act_owl_minus_clean**: top |cos| 0.190 (+) / 0.228 (−); no control

| end | rank | feature | cos | control cos | P | label |
|---|---|---|---|---|---|---|
| + | 1 | 37243 | +0.190 |  |  | sequence \| sequences \| Sequence \| Sequence \| sequence \| series |
| + | 2 | 14546 | +0.156 |  |  | gn \| gni \| emoc \| esac \| reven \| gn |
| + | 3 | 59100 | +0.154 |  |  | in \| to \| for \| at \| of \| and |
| + | 4 | 15145 | +0.122 |  |  | ninth \| seventh \| fourth \| cmp \| fifth \| isions |
| + | 5 | 12969 | +0.116 |  |  | DISTINCT \| CppMethodIntialized \| dissoci \| Nap \| Mockito \| Alibaba |
| − | 1 | 31726 | -0.228 |  |  | shemale \| preferredStyle \| CppCodeGenWriteBarrier \| analsex \| shepherd \| useRalative |
| − | 2 | 58364 | -0.180 |  |  | webElementX \| PostalCodesNL \| ForCanBeConvertedToF \| ForCanBeConverted \| useRalative \| useRal |
| − | 3 | 65483 | -0.176 |  |  | datingsider \| useRalative \| ForCanBeConvertedToF \| ForCanBeConverted \| datingside \| useRal |
| − | 4 | 34597 | -0.161 |  |  | departureday \| JSBracketAccess \| stantiateViewController \| jusqu \| pushViewController \| webElementProperties |
| − | 5 | 11551 | -0.159 |  |  | componentDid \| ishops \| typeparam \| ovolta \| septembre \| safeguards |

**teacher_dolphin**: top |cos| 0.300 (+) / 0.288 (−); no control

| end | rank | feature | cos | control cos | P | label |
|---|---|---|---|---|---|---|
| + | 1 | 31968 | +0.300 |  |  | migrationBuilder \| birka \| najb \| EMPLARY \| kvinde \| NewUrlParser |
| + | 2 | 43539 | +0.208 |  |  | species \| species \| Species \| habitats \| habitat \| Species |
| + | 3 | 42841 | +0.205 |  |  | TextEditingController \| preferably \| techniques \| meticulously \| permite \| spp |
| + | 4 | 22266 | +0.187 |  |  | animal \| animals \| animal \| Animal \| pets \| Animal |
| + | 5 | 25701 | +0.183 |  |  | marine \| ocean \| seaw \| maritime \| sea \| coastal |
| − | 1 | 59100 | -0.288 |  |  | in \| to \| for \| at \| of \| and |
| − | 2 | 29079 | -0.238 |  |  | dataset \| functions \| stunt \| configuration \| config \| remainder |
| − | 3 | 63991 | -0.210 |  |  | iNdEx \| izophren \| dejting \| ederland \| nettsteder \| lesbische |
| − | 4 | 11259 | -0.165 |  |  | ubernetes \| puede \| direccion \| wooded \| kvinde \| Transaction |
| − | 5 | 34238 | -0.163 |  |  | HasBeenSet \| AI \| Sudoku \| metres \| authors \| ActivityCreated |

## Case 2: harmful − benign held-out bundles (mean of unit rows)

### Layer 10

**grad** (80 harmful, 74 benign; |mean harmful − mean benign| = 0.422; shuffled null top-1 p95 0.126, n 200): pass 0/20 (+), 4/20 (−)

| end | rank | feature | cos | P | label |
|---|---|---|---|---|---|
| + (harmful) | 1 | 60973 | +0.118 |  | mortal \| glanced \| statistics \| sands \| ONE \| enough |
| + (harmful) | 2 | 28278 | +0.114 |  | sv \| ish \| episode \| doc \| movie \| Storage |
| + (harmful) | 3 | 41098 | +0.106 |  | Thu \| warranties \| foreach \| Thu \| Fri \| Prest |
| + (harmful) | 4 | 41525 | +0.093 |  | glyph \| occasion \| graphene \| touchdown \| spherical \| journalist |
| + (harmful) | 5 | 48296 | +0.093 |  | Actually \| sabotage \| reassuring \| anson \| detr \| detained |
| − (benign) | 1 | 16575 | -0.140 | P | ld \| Reference \| Reference \| Establishment \| reference \| Navigate |
| − (benign) | 2 | 5009 | -0.132 | P | graf \| MANY \| dbg \| isbn \| WCS \| ad |
| − (benign) | 3 | 42429 | -0.128 | P | duplex \| Output \| txt \| ied \| elif \| ld |
| − (benign) | 4 | 18884 | -0.126 | P | antha \| Lilly \| Zimbabwe \| SharedPreferences \| Fran \| Kansas |
| − (benign) | 5 | 62983 | -0.112 |  | Wa \| Metallic \| Whether \| metallic \| Allied \| abbreviation |

**act** (80 harmful, 74 benign; |mean harmful − mean benign| = 0.066; shuffled null top-1 p95 0.345, n 200): pass 0/20 (+), 0/20 (−)

| end | rank | feature | cos | P | label |
|---|---|---|---|---|---|
| + (harmful) | 1 | 52729 | +0.205 |  | lacking \| lee \| Marc \| lacked \| more \| accompanies |
| + (harmful) | 2 | 57118 | +0.204 |  | but \| yyyy \| ening \| circ \| bound \| spectacular |
| + (harmful) | 3 | 37383 | +0.203 |  | mente \| nie \| aunch \| finer \| membres \| atak |
| + (harmful) | 4 | 50856 | +0.200 |  | udes \| likely \| Palestine \| except \| except \| enduring |
| + (harmful) | 5 | 25098 | +0.198 |  | soever \| rl \| intriguing \| oretical \| nutshell \| numberOf |
| − (benign) | 1 | 7231 | -0.215 |  | UNS \| Prices \| appDelegate \| aunque \| pickerView \| Chronicles |
| − (benign) | 2 | 15682 | -0.213 |  | Level \| Somehow \| misc \| Type \| Okay \| Del |
| − (benign) | 3 | 611 | -0.203 |  | personnel \| emergency \| PriorityQueue \| POL \| Emergency \| emergency |
| − (benign) | 4 | 29014 | -0.193 |  | BracketAccess \| CppGeneric \| ensures \| externalActionCode \| Cumhurba \| SmartyHeaderCode |
| − (benign) | 5 | 59306 | -0.188 |  | ForCanBeConvertedToF \| PostalCodesNL \| ForCanBeConverted \| useRalative \| useRal \| thuisontvangst |

Features in both the gradient and the activation top 40: 0.

### Layer 20

**grad** (80 harmful, 74 benign; |mean harmful − mean benign| = 0.368; shuffled null top-1 p95 0.167, n 200): pass 0/20 (+), 0/20 (−)

| end | rank | feature | cos | P | label |
|---|---|---|---|---|---|
| + (harmful) | 1 | 40420 | +0.114 |  | PowerPoint \| Multi \| ten \| vertically \| always \| nal |
| + (harmful) | 2 | 62857 | +0.114 |  | QT \| Ancient \| crypto \| defective \| allegedly \| blonde |
| + (harmful) | 3 | 37146 | +0.110 |  | op \| zp \| you \| trials \| says \| isin |
| + (harmful) | 4 | 30994 | +0.106 |  | matrix \| Kun \| English \| tip \| nee \| Urban |
| + (harmful) | 5 | 19393 | +0.103 |  | pliers \| Handbook \| handbook \| wirk \| lots \| Maher |
| − (benign) | 1 | 38341 | -0.093 |  | maintains \| objectManager \| ras \| However \| som \| Christina |
| − (benign) | 2 | 56063 | -0.089 |  | purported \| atrib \| harmful \| tributes \| possibile \| may |
| − (benign) | 3 | 26514 | -0.088 |  | usted \| Navigator \| incontri \| noDB \| Making \| regon |
| − (benign) | 4 | 32488 | -0.078 |  | unsafe \| hazardous \| prohibited \| safety \| unsafe \| dangerous |
| − (benign) | 5 | 32991 | -0.075 |  | Javascript \| photo \| Yi \| succesfully \| viagra \| Earl |

**act** (80 harmful, 74 benign; |mean harmful − mean benign| = 0.073; shuffled null top-1 p95 0.345, n 200): pass 0/20 (+), 0/20 (−)

| end | rank | feature | cos | P | label |
|---|---|---|---|---|---|
| + (harmful) | 1 | 33888 | +0.188 |  | But \| But \| but \| yet \| Therefore \| Therefore |
| + (harmful) | 2 | 22760 | +0.187 |  | quelques \| mildly \| minor \| harmless \| gently \| mild |
| + (harmful) | 3 | 10202 | +0.183 |  | except \| meisten \| normal \| normal \| either \| Normal |
| + (harmful) | 4 | 39062 | +0.183 |  | fairly \| somewhat \| considerable \| ish \| quite \| it |
| + (harmful) | 5 | 124 | +0.179 |  | lessly \| iating \| ologically \| ily \| eenth \| rogen |
| − (benign) | 1 | 29818 | -0.219 |  | if \| unless \| calmly \| immediately \| If \| immediate |
| − (benign) | 2 | 46602 | -0.215 |  | response \| worsening \| arrival \| immediate \| swiftly \| decision |
| − (benign) | 3 | 24762 | -0.203 |  | avoid \| Avoid \| Incontri \| discourage \| If \| suspicious |
| − (benign) | 4 | 2119 | -0.203 |  | jika \| unpredictable \| Manufacturer \| randomized \| monary \| illaume |
| − (benign) | 5 | 367 | -0.200 |  | Iter \| gdy \| MR \| dood \| Gil \| etc |

Features in both the gradient and the activation top 40: 0.

### Layer 30

**grad** (80 harmful, 74 benign; |mean harmful − mean benign| = 0.179; shuffled null top-1 p95 0.397, n 200): pass 0/20 (+), 0/20 (−)

| end | rank | feature | cos | P | label |
|---|---|---|---|---|---|
| + (harmful) | 1 | 28469 | +0.140 |  | Erotische \| MessageLookup \| ReferentialAction \| ForCanBeConvertedToForeach \| others \| otherwise |
| + (harmful) | 2 | 13495 | +0.121 |  | bizarre \| ridiculous \| absurd \| weird \| useless \| confusing |
| + (harmful) | 3 | 38842 | +0.119 |  | stuff \| utterly \| indispensable \| chez \| peril \| weird |
| + (harmful) | 4 | 27631 | +0.118 |  | minor \| minor \| Minor \| Minor \| trivial \| mild |
| + (harmful) | 5 | 32571 | +0.117 |  | simply \| just \| merely \| simplement \| just \| Simply |
| − (benign) | 1 | 56082 | -0.121 |  | jedoch \| GCBO \| ONLY \| Please \| externalActionCode \| requested |
| − (benign) | 2 | 35586 | -0.120 |  | presumably \| mentioned \| possibly \| reportedly \| related \| allegedly |
| − (benign) | 3 | 41314 | -0.113 |  | currently \| Util \| pursuant \| June \| Currently \| utilized |
| − (benign) | 4 | 21934 | -0.108 |  | Please \| hereby \| please \| Therefore \| Please \| therefore |
| − (benign) | 5 | 18187 | -0.107 |  | likely \| mention \| clearly \| indicates \| implies \| imply |

**act** (80 harmful, 74 benign; |mean harmful − mean benign| = 0.132; shuffled null top-1 p95 0.329, n 200): pass 0/20 (+), 0/20 (−)

| end | rank | feature | cos | P | label |
|---|---|---|---|---|---|
| + (harmful) | 1 | 27631 | +0.253 |  | minor \| minor \| Minor \| Minor \| trivial \| mild |
| + (harmful) | 2 | 1076 | +0.237 |  | unlikely \| Generally \| generally \| overall \| overall \| aucun |
| + (harmful) | 3 | 21362 | +0.200 |  | adorable \| cozy \| interesting \| playful \| bustling \| nicely |
| + (harmful) | 4 | 17811 | +0.184 |  | usually \| usually \| Usually \| generally \| resolves \| resolve |
| + (harmful) | 5 | 4867 | +0.184 |  | occasional \| quelques \| occasionally \| saja \| few \| simple |
| − (benign) | 1 | 59350 | -0.271 |  | Hannity \| veya \| vehicles \| unsafe \| SOEVER \| atau |
| − (benign) | 2 | 41128 | -0.252 |  | severe \| coma \| dangerously \| rapidly \| death \| fatalities |
| − (benign) | 3 | 56082 | -0.244 |  | jedoch \| GCBO \| ONLY \| Please \| externalActionCode \| requested |
| − (benign) | 4 | 6351 | -0.240 |  | ologic \| may \| may \| medications \| butto \| usually |
| − (benign) | 5 | 23968 | -0.233 |  | emergencies \| emergency \| emergency \| urgent \| Emergency \| Emergency |

Features in both the gradient and the activation top 40: 9 (3713, 12051, 21934, 27631, 28469, 56082, 59350, 59394, 59748).
