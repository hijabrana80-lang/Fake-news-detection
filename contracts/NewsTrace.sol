// SPDX-License-Identifier: MIT
pragma solidity ^0.8.20;

/**
 * @title NewsTrace
 * @notice Anchors a SHA-256 fingerprint of a news article so anyone can prove
 *         that a given text existed at a given time and was reported by a
 *         given address.
 *
 * The hash is passed in as bytes32. The application computes
 * `sha256(normalised_article_text)` off-chain and only stores the digest,
 * which keeps gas cost flat regardless of article length and avoids putting
 * copyrighted content on chain.
 */
contract NewsTrace {
    struct Record {
        address reporter;
        uint256 timestamp;
        string sourceUrl;
        bool exists;
    }

    mapping(bytes32 => Record) private records;

    event NewsRecorded(
        bytes32 indexed contentHash,
        address indexed reporter,
        uint256 timestamp,
        string sourceUrl
    );

    /// @notice Store or refresh the anchor for `contentHash`.
    function record(bytes32 contentHash, string calldata sourceUrl) external {
        records[contentHash] = Record({
            reporter: msg.sender,
            timestamp: block.timestamp,
            sourceUrl: sourceUrl,
            exists: true
        });

        emit NewsRecorded(contentHash, msg.sender, block.timestamp, sourceUrl);
    }

    /// @notice Read an anchor back. Returns found=false when unknown.
    function verify(bytes32 contentHash)
        external
        view
        returns (bool found, address reporter, uint256 timestamp, string memory sourceUrl)
    {
        Record storage entry = records[contentHash];
        return (entry.exists, entry.reporter, entry.timestamp, entry.sourceUrl);
    }

    /// @notice Convenience helper: does this content hash exist on chain?
    function exists(bytes32 contentHash) external view returns (bool) {
        return records[contentHash].exists;
    }
}
