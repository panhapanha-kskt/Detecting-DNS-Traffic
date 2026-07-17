#!/usr/bin/env python3
"""
DNS Feature Extractor - Enhanced for Week 3 Capstone Project
Specifically designed to detect patterns from the DNS attack tool
"""

import math
from collections import Counter, defaultdict
import numpy as np
import re
from datetime import datetime, timedelta
from typing import Dict, List, Tuple, Any, Optional
import csv
import glob
import os
import json
import hashlib

class DNSFeatureExtractor:
    def __init__(self):
        self.query_frequency = defaultdict(int)
        self.nxdomain_count = defaultdict(int)
        self.total_queries = defaultdict(int)
        self.domain_stats = defaultdict(lambda: {'count': 0, 'first_seen': None, 'last_seen': None})
        self.attack_patterns = {
            'nxdomain_flood': 0,
            'random_subdomains': 0,
            'amplification': 0,
            'cache_poisoning': 0,
            'high_entropy': 0,
            'encoding_patterns': 0
        }
        
    def calculate_entropy(self, domain: str) -> float:
        """
        Calculate Shannon entropy of domain name
        Attack tool generates high entropy (>4.0) domains
        """
        if not domain or domain == '':
            return 0.0
        
        entropy = 0.0
        length = len(domain)
        
        # Count frequency of each character
        char_freq = {}
        for char in domain:
            char_freq[char] = char_freq.get(char, 0) + 1
        
        # Calculate entropy
        for count in char_freq.values():
            probability = count / length
            entropy -= probability * math.log2(probability)
            
        return round(entropy, 4)

    def calculate_subdomain_depth(self, domain: str) -> int:
        """
        Count number of subdomain levels
        Attack tool uses deep subdomains (>5) for detection evasion
        """
        if not domain:
            return 0
        
        # Remove trailing dot if present
        domain = domain.rstrip('.')
        
        # Count dots to get depth
        depth = domain.count('.')
        
        return depth

    def calculate_query_length(self, domain: str) -> int:
        """
        Calculate total length of domain name
        Attack tool generates long queries (>100 chars)
        """
        if not domain:
            return 0
        
        # Remove trailing dot
        domain = domain.rstrip('.')
        
        return len(domain)

    def detect_nxdomain_flood(self, domain: str, response_code: str) -> bool:
        """
        Detect NXDOMAIN flood attack patterns
        Attack tool generates nonexistent-XXXXX domains
        """
        if response_code.upper() == 'NXDOMAIN':
            # Check for attack tool pattern: "nxdomain-XXXXXXXX"
            if 'nxdomain-' in domain.lower():
                self.attack_patterns['nxdomain_flood'] += 1
                return True
            
            # Check for random-looking non-existent domains
            labels = domain.split('.')
            if len(labels) > 0 and len(labels[0]) > 10:
                # High entropy on first label might indicate generated NXDOMAIN
                first_label_entropy = self.calculate_entropy(labels[0])
                if first_label_entropy > 3.5:
                    self.attack_patterns['nxdomain_flood'] += 1
                    return True
        
        return False

    def detect_random_subdomains(self, domain: str) -> Tuple[bool, int]:
        """
        Detect random subdomain generation (--random-subdomains)
        Attack tool generates sub1.sub2.sub3... patterns
        """
        labels = domain.split('.')
        
        # Check for deep subdomain structure (attack tool uses 2-8 levels)
        if len(labels) >= 4:
            # Check if subdomains look random (high entropy, alphanumeric)
            random_count = 0
            for i in range(len(labels) - 1):  # Skip TLD
                label = labels[i]
                if len(label) >= 5 and len(label) <= 15:
                    entropy = self.calculate_entropy(label)
                    numeric_ratio = sum(c.isdigit() for c in label) / len(label)
                    
                    # Random subdomains have high entropy and mix of letters/numbers
                    if entropy > 3.0 and 0.2 <= numeric_ratio <= 0.8:
                        random_count += 1
            
            if random_count >= 2:
                self.attack_patterns['random_subdomains'] += 1
                return True, random_count
        
        return False, 0

    def detect_amplification_attack(self, query_type: str, query_length: int, 
                                   edns0_buffer: bool = False) -> bool:
        """
        Detect DNS amplification attempts (--amplification)
        Attack tool uses ANY queries with large EDNS0 buffer
        """
        if query_type.upper() == 'ANY' or query_type == '255':
            # ANY queries are often used in amplification attacks
            self.attack_patterns['amplification'] += 1
            return True
        
        if edns0_buffer or query_length > 512:  # Standard DNS limit is 512 bytes
            self.attack_patterns['amplification'] += 1
            return True
        
        return False

    def detect_cache_poisoning(self, domain: str, timestamps: List[datetime], 
                              transaction_ids: List[int]) -> bool:
        """
        Detect cache poisoning attempts (--cache-poison)
        Attack tool sends multiple queries with different transaction IDs
        """
        if len(timestamps) >= 5:  # Multiple queries in short time
            time_span = (max(timestamps) - min(timestamps)).total_seconds()
            if time_span < 1.0:  # Within 1 second
                # Check for varying transaction IDs
                if len(set(transaction_ids)) >= 3:
                    self.attack_patterns['cache_poisoning'] += 1
                    return True
        
        return False

    def detect_encoding_patterns(self, domain: str) -> Dict[str, Any]:
        """
        Detect base32/base64 encoding patterns (--encoding)
        Attack tool generates .base32.lab.local and .base64.lab.local domains
        """
        result = {
            'is_base32': False,
            'is_base64': False,
            'encoding_type': None,
            'confidence': 0.0
        }
        
        # Direct attack tool pattern detection
        if '.base32.' in domain.lower():
            result['is_base32'] = True
            result['encoding_type'] = 'base32_attack_tool'
            result['confidence'] = 1.0
            self.attack_patterns['encoding_patterns'] += 1
            return result
        
        if '.base64.' in domain.lower():
            result['is_base64'] = True
            result['encoding_type'] = 'base64_attack_tool'
            result['confidence'] = 1.0
            self.attack_patterns['encoding_patterns'] += 1
            return result
        
        # Detect base32 patterns (A-Z, 2-7, length multiple of 8)
        domain_clean = domain.replace('.', '').upper()
        if len(domain_clean) >= 16:
            base32_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZ234567')
            base32_count = sum(1 for c in domain_clean if c in base32_chars)
            base32_ratio = base32_count / len(domain_clean)
            
            if base32_ratio > 0.9 and len(domain_clean) % 8 in [0, 4]:
                result['is_base32'] = True
                result['encoding_type'] = 'base32_suspicious'
                result['confidence'] = base32_ratio
                self.attack_patterns['encoding_patterns'] += 1
        
        # Detect base64 patterns (A-Z, a-z, 0-9, +, /)
        base64_chars = set('ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789+/')
        base64_count = sum(1 for c in domain_clean if c in base64_chars)
        base64_ratio = base64_count / len(domain_clean)
        
        if base64_ratio > 0.95 and (domain_clean.endswith('=') or len(domain_clean) % 4 == 0):
            result['is_base64'] = True
            result['encoding_type'] = 'base64_suspicious'
            result['confidence'] = base64_ratio
            self.attack_patterns['encoding_patterns'] += 1
        
        return result

    def detect_high_entropy_domains(self, domain: str, entropy: float) -> bool:
        """
        Detect high entropy domains (--entropy)
        Attack tool generates domains with entropy > 4.0
        """
        if entropy > 4.0:
            # Check if it's the attack tool's pattern
            labels = domain.split('.')
            if len(labels) >= 2 and len(labels[0]) >= 20:
                # High entropy, long first label is attack tool signature
                self.attack_patterns['high_entropy'] += 1
                return True
        return False

    def calculate_numeric_ratio(self, domain: str) -> float:
        """
        Calculate ratio of numeric characters in domain
        Attack tool uses mix of letters and numbers
        """
        if not domain or len(domain) == 0:
            return 0.0
        
        # Remove dots for calculation
        domain_clean = domain.replace('.', '')
        if len(domain_clean) == 0:
            return 0.0
        
        numeric_count = sum(c.isdigit() for c in domain_clean)
        return round(numeric_count / len(domain_clean), 4)

    def calculate_consecutive_pattern_score(self, domain: str) -> float:
        """
        Detect consecutive character patterns
        Attack tool sometimes uses repetitive patterns
        """
        if not domain or len(domain) < 4:
            return 0.0
        
        domain_clean = domain.replace('.', '')
        
        # Check for repeating characters
        max_repeat = 1
        current_repeat = 1
        
        for i in range(1, len(domain_clean)):
            if domain_clean[i] == domain_clean[i-1]:
                current_repeat += 1
                max_repeat = max(max_repeat, current_repeat)
            else:
                current_repeat = 1
        
        return round(max_repeat / len(domain_clean), 4)

    def parse_technitium_log_line(self, line: str) -> Optional[Dict[str, Any]]:
        """
        Parse a single line from Technitium DNS server log format
        Enhanced to extract more attack-relevant fields
        """
        try:
            line = line.strip()
            if not line:
                return None
            
            # Parse timestamp - format: [2026-02-04 13:37:45 Local]
            timestamp_match = re.search(r'\[(.*?)\s+Local\]', line)
            if not timestamp_match:
                return None
            
            timestamp_str = timestamp_match.group(1)
            try:
                timestamp = datetime.strptime(timestamp_str, '%Y-%m-%d %H:%M:%S')
            except:
                timestamp = datetime.now()
            
            # Parse client IP - format: [172.19.0.1:60486]
            ip_match = re.search(r'\[(\d+\.\d+\.\d+\.\d+):(\d+)\]', line)
            client_ip = ip_match.group(1) if ip_match else 'unknown'
            client_port = int(ip_match.group(2)) if ip_match else 0
            
            # Parse protocol
            protocol_match = re.search(r'\[(UDP|TCP)\]', line)
            protocol = protocol_match.group(1) if protocol_match else 'UDP'
            
            # Parse QNAME - format: QNAME: domain.com;
            qname_match = re.search(r'QNAME:\s*([^;]+);', line)
            if not qname_match:
                return None
            domain = qname_match.group(1).strip().lower()
            
            # Parse QTYPE - format: QTYPE: A;
            qtype_match = re.search(r'QTYPE:\s*([^;]+);', line)
            query_type = qtype_match.group(1).strip() if qtype_match else 'unknown'
            
            # Parse QCLASS
            qclass_match = re.search(r'QCLASS:\s*([^;]+);', line)
            query_class = qclass_match.group(1).strip() if qclass_match else 'IN'
            
            # Parse RCODE - format: RCODE: NoError;
            rcode_match = re.search(r'RCODE:\s*([^;]+);', line)
            response_code = rcode_match.group(1).strip() if rcode_match else 'unknown'
            
            # Parse ANSWER size
            answer_match = re.search(r'ANSWER:\s*\[(.*?)\]', line)
            answer_size = len(answer_match.group(1).split(',')) if answer_match and answer_match.group(1) else 0
            
            # Extract transaction ID if present (for cache poisoning detection)
            tid_match = re.search(r'ID:\s*(\d+)', line)
            transaction_id = int(tid_match.group(1)) if tid_match else None
            
            # Clean domain
            domain = domain.rstrip('.')
            
            # Update domain stats
            self.domain_stats[domain]['count'] += 1
            if self.domain_stats[domain]['first_seen'] is None:
                self.domain_stats[domain]['first_seen'] = timestamp
            self.domain_stats[domain]['last_seen'] = timestamp
            
            return {
                'timestamp': timestamp,
                'domain': domain,
                'client_ip': client_ip,
                'client_port': client_port,
                'protocol': protocol,
                'query_type': query_type,
                'query_class': query_class,
                'response_code': response_code,
                'answer_size': answer_size,
                'transaction_id': transaction_id,
                'raw_line': line
            }
            
        except Exception as e:
            return None

    def extract_features_from_technitium_log(self, log_file: str) -> List[Dict[str, Any]]:
        """
        Extract features from Technitium DNS server log file
        """
        features_list = []
        
        try:
            print(f"Processing log file: {log_file}")
            line_count = 0
            parsed_count = 0
            
            # Store recent queries for cache poisoning detection
            recent_queries = defaultdict(lambda: {'timestamps': [], 'transaction_ids': []})
            
            with open(log_file, 'r', encoding='utf-8', errors='ignore') as f:
                for line in f:
                    line_count += 1
                    parsed = self.parse_technitium_log_line(line)
                    
                    if parsed:
                        parsed_count += 1
                        
                        # Extract features for this query
                        features = self.extract_single_domain_features(parsed['domain'])
                        
                        # Add metadata from log
                        features['timestamp'] = parsed['timestamp'].isoformat() if parsed['timestamp'] else None
                        features['client_ip'] = parsed['client_ip']
                        features['client_port'] = parsed['client_port']
                        features['protocol'] = parsed['protocol']
                        features['query_type'] = parsed['query_type']
                        features['response_code'] = parsed['response_code']
                        features['answer_size'] = parsed['answer_size']
                        
                        # Add time-based features
                        features['hour_of_day'] = parsed['timestamp'].hour if parsed['timestamp'] else -1
                        features['day_of_week'] = parsed['timestamp'].weekday() if parsed['timestamp'] else -1
                        
                        # Update frequency stats
                        self.query_frequency[parsed['domain']] += 1
                        self.total_queries[parsed['client_ip']] += 1
                        
                        # Track recent queries for this domain (for cache poisoning detection)
                        if parsed['domain']:
                            recent_queries[parsed['domain']]['timestamps'].append(parsed['timestamp'])
                            if parsed['transaction_id']:
                                recent_queries[parsed['domain']]['transaction_ids'].append(parsed['transaction_id'])
                            
                            # Keep only last 20 queries
                            if len(recent_queries[parsed['domain']]['timestamps']) > 20:
                                recent_queries[parsed['domain']]['timestamps'].pop(0)
                                recent_queries[parsed['domain']]['transaction_ids'].pop(0)
                        
                        # Attack detection
                        features['is_nxdomain_flood'] = self.detect_nxdomain_flood(
                            parsed['domain'], parsed['response_code']
                        )
                        
                        is_random_subdomain, random_count = self.detect_random_subdomains(parsed['domain'])
                        features['is_random_subdomain'] = is_random_subdomain
                        features['random_subdomain_count'] = random_count
                        
                        # Estimate if EDNS0 was used (based on answer size)
                        edns0_used = parsed['answer_size'] > 10 or parsed['protocol'] == 'TCP'
                        features['is_amplification'] = self.detect_amplification_attack(
                            parsed['query_type'], len(parsed['raw_line']), edns0_used
                        )
                        
                        # Cache poisoning detection (needs historical data)
                        features['is_cache_poisoning'] = self.detect_cache_poisoning(
                            parsed['domain'],
                            recent_queries[parsed['domain']]['timestamps'],
                            recent_queries[parsed['domain']]['transaction_ids']
                        )
                        
                        # Encoding detection
                        encoding_results = self.detect_encoding_patterns(parsed['domain'])
                        features.update(encoding_results)
                        
                        # High entropy detection
                        features['is_high_entropy'] = self.detect_high_entropy_domains(
                            parsed['domain'], features['entropy']
                        )
                        
                        # Calculate query rate (queries per second for this client)
                        features['query_rate'] = self.query_frequency[parsed['domain']] / 60.0  # Approximate
                        
                        # Add attack signature
                        attack_signature = []
                        if features['is_nxdomain_flood']:
                            attack_signature.append('NXDOMAIN_FLOOD')
                        if features['is_random_subdomain']:
                            attack_signature.append('RANDOM_SUBDOMAIN')
                        if features['is_amplification']:
                            attack_signature.append('AMPLIFICATION')
                        if features['is_cache_poisoning']:
                            attack_signature.append('CACHE_POISONING')
                        if features['is_high_entropy']:
                            attack_signature.append('HIGH_ENTROPY')
                        if features['is_base32'] or features['is_base64']:
                            attack_signature.append('ENCODING')
                        
                        features['attack_signature'] = ','.join(attack_signature) if attack_signature else 'NORMAL'
                        features['attack_count'] = len(attack_signature)
                        
                        features_list.append(features)
            
            print(f"  Completed: {line_count} lines processed, {parsed_count} DNS queries extracted")
            
        except Exception as e:
            print(f"Error reading log file {log_file}: {e}")
        
        return features_list

    def extract_features_from_multiple_logs(self, log_directory: str, pattern: str = "*.log") -> List[Dict[str, Any]]:
        """
        Extract features from multiple Technitium log files in a directory
        """
        all_features = []
        log_files = glob.glob(os.path.join(log_directory, pattern))
        
        # Sort log files by date
        log_files.sort()
        
        print(f"Found {len(log_files)} log files to process")
        
        for log_file in log_files:
            features = self.extract_features_from_technitium_log(log_file)
            all_features.extend(features)
            
            # Add log file source
            for f in features:
                f['source_file'] = os.path.basename(log_file)
        
        return all_features

    def extract_single_domain_features(self, domain: str) -> Dict[str, Any]:
        """
        Extract all features for a single domain
        """
        features = {}
        
        # Basic features
        features['domain'] = domain
        features['entropy'] = self.calculate_entropy(domain)
        features['subdomain_depth'] = self.calculate_subdomain_depth(domain)
        features['query_length'] = self.calculate_query_length(domain)
        features['numeric_ratio'] = self.calculate_numeric_ratio(domain)
        features['consecutive_pattern_score'] = self.calculate_consecutive_pattern_score(domain)
        
        # Extract TLD and main domain
        labels = domain.split('.')
        features['tld'] = labels[-1] if labels else ''
        features['main_domain'] = '.'.join(labels[-2:]) if len(labels) >= 2 else domain
        features['num_labels'] = len(labels)
        
        # First label analysis (often contains the attack payload)
        features['first_label'] = labels[0] if labels else ''
        features['first_label_length'] = len(features['first_label'])
        features['first_label_entropy'] = self.calculate_entropy(features['first_label'])
        
        return features

    def export_features_to_json(self, features_list: List[Dict[str, Any]], output_file: str, pretty: bool = True):
        """
        Export features to JSON file
        """
        if not features_list:
            print("No features to export")
            return
        
        try:
            # Create directory if it doesn't exist
            os.makedirs(os.path.dirname(output_file), exist_ok=True)
            
            # Convert datetime objects to strings if any remain
            json_compatible_list = []
            for feature in features_list:
                feature_copy = feature.copy()
                # Ensure all values are JSON serializable
                for key, value in feature_copy.items():
                    if isinstance(value, (datetime, np.integer, np.floating, np.ndarray)):
                        if isinstance(value, datetime):
                            feature_copy[key] = value.isoformat()
                        else:
                            feature_copy[key] = value.item() if hasattr(value, 'item') else value
                json_compatible_list.append(feature_copy)
            
            with open(output_file, 'w', encoding='utf-8') as f:
                if pretty:
                    json.dump(json_compatible_list, f, indent=2, ensure_ascii=False)
                else:
                    json.dump(json_compatible_list, f, ensure_ascii=False)
            
            print(f"\n✅ Features exported to JSON: {output_file}")
            print(f"   Total records: {len(features_list)}")
            
        except Exception as e:
            print(f"Error exporting to JSON: {e}")

    def export_attack_summary(self, features_list: List[Dict[str, Any]], output_file: str):
        """
        Export attack detection summary
        """
        if not features_list:
            return
        
        # Count attacks by type
        attack_counts = {
            'NXDOMAIN_FLOOD': sum(1 for f in features_list if f.get('is_nxdomain_flood', False)),
            'RANDOM_SUBDOMAIN': sum(1 for f in features_list if f.get('is_random_subdomain', False)),
            'AMPLIFICATION': sum(1 for f in features_list if f.get('is_amplification', False)),
            'CACHE_POISONING': sum(1 for f in features_list if f.get('is_cache_poisoning', False)),
            'HIGH_ENTROPY': sum(1 for f in features_list if f.get('is_high_entropy', False)),
            'ENCODING': sum(1 for f in features_list if f.get('is_base32', False) or f.get('is_base64', False))
        }
        
        # Group by client IP
        ip_attacks = defaultdict(lambda: defaultdict(int))
        for f in features_list:
            if f.get('attack_count', 0) > 0:
                ip = f.get('client_ip', 'unknown')
                for attack in f.get('attack_signature', '').split(','):
                    if attack != 'NORMAL':
                        ip_attacks[ip][attack] += 1
        
        summary = {
            'total_queries': len(features_list),
            'total_attacks': sum(attack_counts.values()),
            'attack_percentage': round(sum(attack_counts.values()) / len(features_list) * 100, 2) if features_list else 0,
            'attack_breakdown': attack_counts,
            'attack_patterns_detected': self.attack_patterns,
            'top_attackers': [
                {
                    'ip': ip,
                    'total_attacks': sum(counts.values()),
                    'breakdown': dict(counts)
                }
                for ip, counts in sorted(ip_attacks.items(), 
                                        key=lambda x: sum(x[1].values()), 
                                        reverse=True)[:10]
            ]
        }
        
        # Save to JSON
        try:
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(summary, f, indent=2, ensure_ascii=False)
            print(f"\n✅ Attack summary exported to: {output_file}")
        except Exception as e:
            print(f"Error exporting attack summary: {e}")
        
        return summary

    def print_feature_summary(self, features_list: List[Dict[str, Any]]):
        """
        Print summary statistics of extracted features with attack detection focus
        """
        if not features_list:
            print("No features to summarize")
            return
        
        print("\n" + "="*80)
        print("DNS FEATURE EXTRACTION SUMMARY - WEEK 3 (ATTACK DETECTION FOCUS)")
        print("="*80)
        print(f"Total queries processed: {len(features_list)}")
        
        # Attack detection summary
        attack_queries = [f for f in features_list if f.get('attack_count', 0) > 0]
        print(f"\n🔴 ATTACK DETECTION SUMMARY:")
        print(f"   Suspicious queries: {len(attack_queries)} ({len(attack_queries)/len(features_list)*100:.1f}%)")
        
        if attack_queries:
            print("\n   Attack Type Breakdown:")
            attack_types = Counter()
            for f in attack_queries:
                for attack in f.get('attack_signature', '').split(','):
                    if attack != 'NORMAL':
                        attack_types[attack] += 1
            
            for attack_type, count in attack_types.most_common():
                print(f"     • {attack_type}: {count}")
        
        # Date range
        timestamps = [f['timestamp'] for f in features_list if f.get('timestamp')]
        if timestamps:
            print(f"\n📅 Date range: {min(timestamps)[:10]} to {max(timestamps)[:10]}")
        
        # Unique domains and IPs
        unique_domains = len(set(f['domain'] for f in features_list))
        unique_ips = len(set(f.get('client_ip', 'unknown') for f in features_list if f.get('client_ip')))
        print(f"   Unique domains: {unique_domains}")
        print(f"   Unique client IPs: {unique_ips}")
        
        # Attack tool specific patterns
        print("\n🎯 ATTACK TOOL PATTERNS DETECTED:")
        print(f"   NXDOMAIN flood patterns: {self.attack_patterns['nxdomain_flood']}")
        print(f"   Random subdomain patterns: {self.attack_patterns['random_subdomains']}")
        print(f"   Amplification attempts: {self.attack_patterns['amplification']}")
        print(f"   Cache poisoning attempts: {self.attack_patterns['cache_poisoning']}")
        print(f"   High entropy domains: {self.attack_patterns['high_entropy']}")
        print(f"   Encoding patterns: {self.attack_patterns['encoding_patterns']}")
        
        # Feature statistics for baseline
        print("\n📊 FEATURE STATISTICS (for baseline establishment):")
        print("-" * 60)
        
        feature_stats = {
            'entropy': [],
            'subdomain_depth': [],
            'query_length': [],
            'numeric_ratio': [],
            'first_label_entropy': [],
            'first_label_length': []
        }
        
        for f in features_list:
            for feature in feature_stats.keys():
                if feature in f and f[feature] is not None:
                    feature_stats[feature].append(f[feature])
        
        for feature, values in feature_stats.items():
            if values:
                print(f"\n{feature}:")
                print(f"  Mean: {np.mean(values):.4f}")
                print(f"  Std: {np.std(values):.4f}")
                print(f"  Min: {min(values):.4f}")
                print(f"  Max: {max(values):.4f}")
        
        # Top suspicious domains
        if attack_queries:
            print("\n⚠  TOP 10 MOST SUSPICIOUS DOMAINS:")
            sorted_suspicious = sorted(attack_queries, 
                                      key=lambda x: (x.get('attack_count', 0), x.get('entropy', 0)), 
                                      reverse=True)[:10]
            for i, domain in enumerate(sorted_suspicious, 1):
                attacks = domain.get('attack_signature', 'NORMAL')
                print(f"  {i}. {domain['domain']}")
                print(f"     Attacks: {attacks} | Entropy: {domain.get('entropy', 0):.2f} | Length: {domain.get('query_length', 0)}")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="DNS Feature Extractor")
    parser.add_argument("--input",  type=str, default=None,
                        help="Single log file to extract (overrides hardcoded log_directory)")
    parser.add_argument("--output", type=str, default=None,
                        help="Output JSON path (overrides hardcoded output_directory)")
    args = parser.parse_args()

    extractor = DNSFeatureExtractor()

    # Paths — CLI args override hardcoded defaults
    log_directory    = "/home/paifern/DNS-_raffic-_Anomaly_Detection/technitium-dns/data/dns_logs"
    output_directory = "/home/paifern/DNS-_raffic-_Anomaly_Detection/data"
    os.makedirs(output_directory, exist_ok=True)

    print("="*80)
    print("DNS FEATURE EXTRACTION FROM TECHNITIUM LOGS - WEEK 3")
    print("Enhanced for Attack Tool Detection")
    print("="*80)

    # Single-file mode (called by Train_Models.sh per-log)
    if args.input:
        print(f"\n📁 Single file mode: {args.input}")
        features = extractor.extract_features_from_technitium_log(args.input)
        out_path = args.output if args.output else os.path.join(
            output_directory,
            "extracted_" + os.path.basename(args.input).replace(".log", ".json")
        )
    else:
        # Original batch mode — scan whole directory
        print(f"\n📁 Log directory: {log_directory}")
        print(f"📁 Output directory: {output_directory}\n")
        features = extractor.extract_features_from_multiple_logs(log_directory, "*.log")
        out_path = args.output if args.output else os.path.join(output_directory, "week3_features_all.json")

    if features:
        extractor.print_feature_summary(features)

        # Always write to the resolved out_path
        extractor.export_features_to_json(features, out_path, pretty=True)

        # Only write the extra summary files in batch (no --input) mode
        if not args.input:
            attack_summary_json = os.path.join(output_directory, "week3_attack_summary.json")
            suspicious_json     = os.path.join(output_directory, "week3_suspicious_domains.json")
            summary_file        = os.path.join(output_directory, "week3_project_summary.json")

            attack_summary = extractor.export_attack_summary(features, attack_summary_json)
            suspicious_domains = [f for f in features if f.get('attack_count', 0) > 0]
            if suspicious_domains:
                extractor.export_features_to_json(suspicious_domains, suspicious_json, pretty=True)
                print(f"\n⚠  Found {len(suspicious_domains)} suspicious domains for Week 4 testing")

            project_summary = {
                'project': 'DNS Traffic Anomaly Detection',
                'week': 3,
                'total_queries': len(features),
                'suspicious_queries': len(suspicious_domains) if suspicious_domains else 0,
                'attack_patterns_detected': extractor.attack_patterns,
            }
            with open(summary_file, 'w') as f:
                json.dump(project_summary, f, indent=2)

            print(f"\n✅ Project summary saved to: {summary_file}")
            print(f"\n📂 Files in output directory:")
            for file in sorted(os.listdir(output_directory)):
                if file.startswith("week3_"):
                    file_path = os.path.join(output_directory, file)
                    size = os.path.getsize(file_path)
                    print(f"  • {file} ({size:,} bytes)")
    else:
        print("\n❌ No features extracted.")
