// SPDX-License-Identifier: UNLICENSED
pragma solidity ^0.8.13;

import "forge-std/Test.sol";
import "../src/Contract.sol";

contract ContractTest is Test {
    SimpleVotedAggregator myContract;

    function logAggScores(SimpleVotedAggregator agg) public view {
        for (uint256 i=0; i<agg.candidateIdCount(); i++) {
            console.log("CandidateID", i, "Score", agg.candidateIdToScore(i));
        }
    }

    function setUp() public {
        SimpleVotedAggregator aggregator = new SimpleVotedAggregator();

        vm.warp(1658108640);
        aggregator.submitCandidate{value: 0.0001 ether}(
            SimpleVotedAggregator.Candidate(0,0, "data:text/html,%3Ch1%3EHello%2C%20World%21%3C%2Fh1%3E")
        );
        vm.warp(1658108640+60*60*24);
        aggregator.submitCandidate{value: 0.0001 ether}(
            SimpleVotedAggregator.Candidate(0,0, "data:text/html,%3Ch1%3EHello%2C%20WorldPart2%21%3C%2Fh1%3E")
        );
        aggregator.recalculateScores();
        logAggScores(aggregator);

        console.log("\nVoting for 0");
        aggregator.submitVote{value: 0.0000 ether}(
            SimpleVotedAggregator.Vote(0, 0)
        );
        aggregator.recalculateScores();
        logAggScores(aggregator);

        console.log("\nVoting for 0 again");
        aggregator.submitVote{value: 0.0000 ether}(
            SimpleVotedAggregator.Vote(0, 0)
        );
        aggregator.recalculateScores();
        logAggScores(aggregator);

        console.log("\nSampling Counts from 100");
        uint[] memory counts = new uint[](aggregator.candidateIdCount());
        for (uint i=0; i<100; i++) {
            vm.roll(i+1);
            SimpleVotedAggregator.Candidate memory c = aggregator.sampleCandidate();
            counts[c.id]++;
            // console.log(aggregator.sampleCandidate().uri);
        }
        for (uint i=0; i<counts.length; i++) {
            console.log("Id", i, "Count", counts[i]);
            console.log("Uri", aggregator.getCandidate(i).uri);
        }
    }

    function testExample() public {
        assertTrue(true);
    }
}
