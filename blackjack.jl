using Cards
using StatsBase

mutable struct Deck
    deck::Hand
end

suits = [♣, ♢, ♡, ♠]
deck = [Card(r,s) for s in suits for r = 2:14]

function suit(c::Card)
    Suit((0x30 & c.value) >>> 4)
end

function rank(c::Card)
    (c.value & 0x0f) % Int8
end

function blackjackSum(hand::Array{Card})
    ranks = [rank(card) for card ∈ hand]
    numAces = length(findall(x -> x ∈ [Card(14,i) for i in 1:4], hand))
    # Returns sum with high aces and low aces
    out = sum(x -> min(10, x), ranks) + numAces, sum(x -> min(10, x), ranks) - 9*numAces
    return out
end

function shuffleDeck(nDecks::Integer)
    return [Card(r,s) for s in suits for r = 2:14 for _ in 1:nDecks]
end

function Deal(deck::Array{Card,1}, cardsPerHand::Integer, numHands::Integer)
    out = Array{Card}(undef, numHands, cardsPerHand)
    for i ∈ 1:numHands
        for j ∈ 1:cardsPerHand
            out[i,j] = rand(deck)
            deleteat!(deck, findall(x->x == out[i,j], deck))
        end
    end
    println(length(deck))
    return out
end

deck = shuffleDeck(1)
hand = Deal(deck, 2, 1)
blackjackSum(hand)