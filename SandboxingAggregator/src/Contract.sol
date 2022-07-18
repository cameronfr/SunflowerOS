// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import "forge-std/console.sol";

contract SimpleVotedAggregator {

    struct Vote {
        uint time;
        uint candidateId;
    }
    struct Candidate {
        uint submitTime;
        uint id;
        string uri;
    }
    mapping(address => Vote) voterAddressToVote;
    mapping(uint => address) voterIdToVoterAddress;
    mapping(address => bool) voterAddressToExists;
    mapping(uint => Candidate) candidateIdToCandidate;
    uint public candidateIdCount;
    uint public voterIdCount;

    struct Score {
        uint combinedScore;
    }
    mapping(uint => Score) public candidateIdToScore;

    function max(int256 a, int256 b) public pure returns (int256) {
        if (a < b) {
            return b;
        } else {
            return a;
        }
    }

    constructor() {
        candidateIdCount = 0;
        voterIdCount = 0;
        console.log("constructor");
    }


    function submitCandidate(Candidate memory c) public payable {
        require(msg.value == 0.0001 ether, "Send correct ether amount.");
        c.submitTime = block.timestamp;
        c.id = candidateIdCount;
        candidateIdToCandidate[candidateIdCount] = c;
        candidateIdCount++;
    }

    function submitVote(Vote memory v) public payable {
        require(msg.value == 0.000 ether, "Send correct ether amount.");
        if (voterAddressToExists[msg.sender] == false) {
            voterAddressToExists[msg.sender] = true;
            voterIdToVoterAddress[voterIdCount] = msg.sender;
            voterIdCount++;
        }
        voterAddressToVote[msg.sender] = v;
    }

    function recalculateScores() public {
        // Zero all scores
        for (uint i=0; i<candidateIdCount; i++) {
            candidateIdToScore[i] = Score(0);
        }
        uint[] memory voteCounts = new uint[](candidateIdCount);
        // Iterate through votes of *all* voters, accumulate to votingScore
        for (uint i=0; i<voterIdCount; i++) {
            Vote storage v = voterAddressToVote[voterIdToVoterAddress[i]];
            voteCounts[v.candidateId]++;
        }
        // Iterate through candidates, calc final score
        for (uint i=0; i<candidateIdCount; i++) {
            Candidate storage c = candidateIdToCandidate[i];
            uint recentnessScore = uint(max((7*24*60*60) - (int256(block.timestamp)-int256(c.submitTime)), 0));
            // Essentiallly recentness normalized to 0-1000 (just added is score 1000, 1 week old is score 0) and #votes normalized to 0-1000 (all votes is 1000, no votes is 0). And weighting votes 5x.
            uint combinedScore = (recentnessScore / 605) + 5*((1000 * voteCounts[i]) / (voterIdCount+1));
            // Normalize to 0-1000 again
            candidateIdToScore[i].combinedScore = combinedScore / 6;
        }

    }

    function sampleCandidate() public view returns (Candidate memory) {
        uint256 totalScoresMaxSum = 1000 * candidateIdCount;
        uint256 semiRand = uint256(blockhash(block.number-1)) % totalScoresMaxSum;
        for (uint i=0; true; i++) {
            uint id = i%candidateIdCount;
            if (semiRand <= candidateIdToScore[id].combinedScore) {
                return candidateIdToCandidate[id];
            } else {
                semiRand = semiRand - candidateIdToScore[id].combinedScore;
            }
        }
    }

    function getCandidate(uint i) public view returns (Candidate memory) {
        return candidateIdToCandidate[i];
    }

}
