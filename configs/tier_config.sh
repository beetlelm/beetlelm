#!/bin/bash

tier_config () {

case $1 in

# ---------- Tier 1 ----------
babylm-zho|babylm-fas|babylm-eng|babylm-nld|babylm-bul|babylm-fra|babylm-ind|babylm-deu|babylm-ukr)
    STEPS=6000
    CKPT=1000
    ;;

# ---------- Tier 2 ----------
babylm-ita|babylm-spa|babylm-jpn|babylm-ara|babylm-ell|babylm-eus|babylm-pol|babylm-cym|babylm-swe|babylm-por|babylm-afr|babylm-est|babylm-srp)
    STEPS=3000
    CKPT=500
    ;;

# ---------- Tier 3 ----------
*)
    STEPS=1200
    CKPT=200
    ;;
esac

echo "$STEPS $CKPT"
}
